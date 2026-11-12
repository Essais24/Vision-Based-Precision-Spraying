// DeepStream 9.1 nvdsvideotemplate custom library.
// Computes ExG, removes pixels in nvinfer object boxes on GPU,
// and extracts hardware optical flow metadata (Gst-nvof).
#include <cuda_runtime.h>
#include <gst/gst.h>
#include <vector>
#include <string>
#include <algorithm>
#include "nvbufsurface.h"
#include "nvdsmeta.h"
#include "gstnvdsmeta.h"
#include "nvds_opticalflow_meta.h"
#include "nvdscustomlib_base.hpp"

struct Rect { int left, top, right, bottom; };


// Exported optical flow state accessible via ctypes
static float g_latest_flow_x = 0.0f;
static float g_latest_flow_y = 0.0f;
static unsigned long long g_flow_frame_count = 0;

extern "C" void get_latest_optical_flow(float *flow_x, float *flow_y) {
  if (flow_x) *flow_x = g_latest_flow_x;
  if (flow_y) *flow_y = g_latest_flow_y;
}

extern "C" unsigned long long get_optical_flow_frame_count() {
  return g_flow_frame_count;
}

/*
 * CUDA Kernel: exg_mask_rgba
 * Computes ExG = 2*G - R - B for RGBA buffers.
 * Avoids pink tint and channel-swap artifacts by explicitly writing all four channels
 * (R, G, B, A=255) when in mask mode, or using an explicit overlay marker.
 *
 * mode:
 *   0: Legacy carrier: pure magenta (R=255, G=0, B=255, A=255) for weeds, original pixel otherwise.
 *   1: Binary mask: R=G=B=255, A=255 for weeds; R=G=B=0, A=255 for background.
 *   2: Balanced green overlay: R=0, G=255, B=0, A=255 for weeds, original pixel otherwise.
 */
__global__ void exg_mask_rgba(uchar4 *image, int width, int height, int pitch,
                              const Rect *excluded, int count, int threshold,
                              int mode) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;

  uchar4 *pixel = reinterpret_cast<uchar4 *>(reinterpret_cast<unsigned char *>(image) + y * pitch) + x;
  const uchar4 original = *pixel;

  bool inside = false;
  for (int i = 0; i < count; ++i) {
    const Rect r = excluded[i];
    inside |= (x >= r.left && x < r.right && y >= r.top && y < r.bottom);
  }

  // RGBA memory layout: x=R, y=G, z=B, w=A
  const int exg = 2 * int(original.y) - int(original.x) - int(original.z);
  const bool is_weed = (!inside && exg > threshold);

  if (mode == 1) {
    // Mode 1: Explicit binary mask where all RGB channels match (R=G=B=val, A=255).
    // Prevents any uninitialized or stale R/B channels causing pink/magenta artifacts.
    const unsigned char val = is_weed ? 255 : 0;
    *pixel = make_uchar4(val, val, val, 255);
  } else if (mode == 2) {
    // Mode 2: Clean green overlay for preview (R=0, G=255, B=0, A=255).
    if (is_weed) {
      *pixel = make_uchar4(0, 255, 0, 255);
    } else {
      *pixel = make_uchar4(original.x, original.y, original.z, 255);
    }
  } else {
    // Mode 0: Legacy carrier: magenta marker (R=255, G=0, B=255, A=255).
    if (is_weed) {
      *pixel = make_uchar4(255, 0, 255, 255);
    } else {
      *pixel = make_uchar4(original.x, original.y, original.z, 255);
    }
  }
}

class WeedMaskAlgorithm final : public DSCustomLibraryBase {
 public:
  bool SetInitParams(DSCustom_CreateParams *params) override {
    if (!DSCustomLibraryBase::SetInitParams(params)) return false;
    stream_ = params->m_cudaStream;
    return cudaSetDevice(params->m_gpuId) == cudaSuccess;
  }

  bool SetProperty(Property &prop) override {
    if (prop.key == "exg-threshold") {
      threshold_ = std::clamp(std::stoi(prop.value), -510, 510);
      return true;
    }
    if (prop.key == "exg-mode") {
      mode_ = std::clamp(std::stoi(prop.value), 0, 2);
      return true;
    }
    GST_WARNING_OBJECT(m_element, "Unknown weed template property: %s", prop.key.c_str());
    return false;
  }

  bool HandleEvent(GstEvent *) override { return true; }
  char *QueryProperties() override { return nullptr; }

  GstCaps *GetCompatibleCaps(GstPadDirection, GstCaps *in_caps, GstCaps *) override {
    // Strictly enforce RGBA NVMM memory caps across both sink and src pads.
    return gst_caps_copy(in_caps);
  }

  BufferResult ProcessBuffer(GstBuffer *buffer) override {
    NvBufSurface *surface = getNvBufSurface(buffer);
    NvDsBatchMeta *batch = gst_buffer_get_nvds_batch_meta(buffer);
    if (!surface || !batch) return BufferResult::Buffer_Error;

    for (NvDsMetaList *node = batch->frame_meta_list; node; node = node->next) {
      auto *frame = static_cast<NvDsFrameMeta *>(node->data);
      if (frame->batch_id >= surface->batchSize) continue;
      NvBufSurfaceParams &s = surface->surfaceList[frame->batch_id];
      if (s.colorFormat != NVBUF_COLOR_FORMAT_RGBA || !s.dataPtr) {
        GST_ERROR_OBJECT(m_element, "weed template requires RGBA NVMM input");
        return BufferResult::Buffer_Error;
      }

      // Check for hardware optical flow metadata (Gst-nvof) attached to this frame
      for (NvDsMetaList *l_user = frame->frame_user_meta_list; l_user != nullptr; l_user = l_user->next) {
        auto *user_meta = static_cast<NvDsUserMeta *>(l_user->data);
        if (user_meta && user_meta->base_meta.meta_type == NVDS_OPTICAL_FLOW_META) {
          auto *of_meta = static_cast<NvDsOpticalFlowMeta *>(user_meta->user_meta_data);
          if (of_meta && of_meta->data && of_meta->rows > 0 && of_meta->cols > 0) {
            auto *flow = static_cast<const NvOFFlowVector *>(of_meta->data);
            long long sum_x = 0;
            long long sum_y = 0;
            const size_t total_vectors = static_cast<size_t>(of_meta->rows) * of_meta->cols;
            for (size_t k = 0; k < total_vectors; ++k) {
              sum_x += flow[k].flowx;
              sum_y += flow[k].flowy;
            }
            g_latest_flow_x = static_cast<float>(sum_x) / static_cast<float>(total_vectors);
            g_latest_flow_y = static_cast<float>(sum_y) / static_cast<float>(total_vectors);
            g_flow_frame_count++;
          }
        }
      }

      std::vector<Rect> host_rects;
      for (NvDsMetaList *object = frame->obj_meta_list; object; object = object->next) {
        auto *meta = static_cast<NvDsObjectMeta *>(object->data);
        const NvOSD_RectParams &r = meta->rect_params;
        host_rects.push_back({
          std::max(0, int(r.left)), std::max(0, int(r.top)),
          std::min(int(s.width), int(r.left + r.width)),
          std::min(int(s.height), int(r.top + r.height)),
        });
      }

      Rect *device_rects = nullptr;
      if (!host_rects.empty()) {
        if (cudaMalloc(&device_rects, host_rects.size() * sizeof(Rect)) != cudaSuccess ||
            cudaMemcpyAsync(device_rects, host_rects.data(), host_rects.size() * sizeof(Rect),
                            cudaMemcpyHostToDevice, stream_) != cudaSuccess) {
          if (device_rects) cudaFree(device_rects);
          return BufferResult::Buffer_Error;
        }
      }

      const dim3 block(32, 8);
      const dim3 grid((s.width + block.x - 1) / block.x, (s.height + block.y - 1) / block.y);
      exg_mask_rgba<<<grid, block, 0, stream_>>>(static_cast<uchar4 *>(s.dataPtr), s.width,
          s.height, s.pitch, device_rects, int(host_rects.size()), threshold_, mode_);
      const cudaError_t status = cudaGetLastError();
      if (device_rects) cudaFree(device_rects);
      if (status != cudaSuccess) {
        GST_ERROR_OBJECT(m_element, "weed ExG CUDA launch failed: %s", cudaGetErrorString(status));
        return BufferResult::Buffer_Error;
      }

    }
    return BufferResult::Buffer_Ok;
  }

 private:
  cudaStream_t stream_ = nullptr;
  int threshold_ = 20;
  int mode_ = 0;
};

extern "C" IDSCustomLibrary *CreateCustomAlgoCtx(DSCustom_CreateParams *) {
  return new WeedMaskAlgorithm();
}
