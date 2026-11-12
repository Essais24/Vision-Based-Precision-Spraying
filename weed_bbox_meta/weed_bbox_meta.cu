// ExG mask generation and weed object metadata insertion before NvSORT.
#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <opencv2/imgproc.hpp>
#include <gst/gst.h>
#include <gstnvdsmeta.h>
#include <nvbufsurface.h>
#include <nvdscustomlib_base.hpp>

namespace {
constexpr int kWeedClassId = 1;  // YOLO chilli detections use class 0
constexpr int kDefaultMinArea = 3000;
constexpr int kDefaultMaxBoxArea = 100000;
constexpr int kMaxWeedsPerFrame = 200;
constexpr int kMaxWatershedSeeds = 80;

void add_component_boxes(const cv::Mat &labels, int label, const cv::Rect &bounds,
                         int area, int min_area, int cluster_area,
                         int separation_px, std::vector<cv::Rect> &boxes) {
  const cv::Rect fallback(bounds.x, bounds.y, bounds.width, bounds.height);
  auto add_fallback = [&]() {
    if (area >= min_area) boxes.push_back(fallback);
  };
  if (area <= cluster_area) {
    add_fallback();
    return;
  }

  cv::Mat component;
  cv::compare(labels(bounds), cv::Scalar(label), component, cv::CMP_EQ);
  component *= 255;
  cv::Mat distance;
  cv::distanceTransform(component, distance, cv::DIST_L2, 5);
  double max_distance = 0.0;
  cv::minMaxLoc(distance, nullptr, &max_distance);
  if (max_distance <= 0.0) {
    add_fallback();
    return;
  }

  separation_px = std::max(1, separation_px);
  if (separation_px % 2 == 0) ++separation_px;
  cv::Mat kernel = cv::getStructuringElement(
      cv::MORPH_ELLIPSE, cv::Size(separation_px, separation_px));
  cv::Mat dilated, local_max, above_floor, seeds;
  cv::dilate(distance, dilated, kernel);
  cv::compare(distance, dilated, local_max, cv::CMP_EQ);
  cv::compare(distance, cv::Scalar(0.15 * max_distance), above_floor, cv::CMP_GT);
  cv::bitwise_and(local_max, above_floor, seeds);

  cv::Mat seed_labels;
  const int seed_count = cv::connectedComponents(seeds, seed_labels, 8, CV_32S);
  if (seed_count <= 2) {
    add_fallback();
    return;
  }
  if (seed_count - 1 > kMaxWatershedSeeds) return;

  cv::Mat unknown, markers = seed_labels + 1, color_component;
  cv::subtract(component, seeds, unknown);
  cv::Mat unknown_pixels;
  cv::compare(unknown, cv::Scalar(255), unknown_pixels, cv::CMP_EQ);
  markers.setTo(cv::Scalar(0), unknown_pixels);
  cv::cvtColor(component, color_component, cv::COLOR_GRAY2BGR);
  cv::watershed(color_component, markers);

  double max_marker = 0.0;
  cv::minMaxLoc(markers, nullptr, &max_marker);
  bool added = false;
  for (int marker = 2; marker <= static_cast<int>(max_marker); ++marker) {
    cv::Mat region;
    cv::compare(markers, cv::Scalar(marker), region, cv::CMP_EQ);
    if (cv::countNonZero(region) < min_area) continue;
    cv::Rect box = cv::boundingRect(region);
    box.x += bounds.x;
    box.y += bounds.y;
    boxes.push_back(box);
    added = true;
  }
  if (!added) add_fallback();
}

__global__ void make_exg_mask(const uchar4 *rgba, size_t rgba_pitch,
                              uint8_t *mask, int width, int height,
                              int threshold) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;
  const uchar4 pixel = *reinterpret_cast<const uchar4 *>(
      reinterpret_cast<const char *>(rgba) + y * rgba_pitch + x * sizeof(uchar4));
  const int exg = 2 * static_cast<int>(pixel.y) -
                  static_cast<int>(pixel.x) - static_cast<int>(pixel.z);
  mask[y * width + x] = exg > threshold ? 255 : 0;
}

class WeedBboxMeta final : public DSCustomLibraryBase {
 public:
  ~WeedBboxMeta() override {
    if (device_mask_) cudaFree(device_mask_);
  }

  bool SetInitParams(DSCustom_CreateParams *params) override {
    return DSCustomLibraryBase::SetInitParams(params);
  }

  bool SetProperty(Property &prop) override {
    try {
      if (prop.key == "exg-threshold") threshold_ = std::stoi(prop.value);
      else if (prop.key == "min-area") min_area_ = std::max(1, std::stoi(prop.value));
      else if (prop.key == "max-box-area") max_box_area_ = std::max(0, std::stoi(prop.value));
      else if (prop.key == "cluster-area") cluster_area_ = std::max(1, std::stoi(prop.value));
      else if (prop.key == "separation-px") separation_px_ = std::max(1, std::stoi(prop.value));
    } catch (...) {
      GST_ERROR("weed-bbox-meta: invalid property %s=%s", prop.key.c_str(), prop.value.c_str());
      return false;
    }
    return true;
  }

  bool HandleEvent(GstEvent *) override { return true; }
  char *QueryProperties() override { return nullptr; }

  BufferResult ProcessBuffer(GstBuffer *buffer) override {
    GstMapInfo map = GST_MAP_INFO_INIT;
    if (!gst_buffer_map(buffer, &map, GST_MAP_READ)) return BufferResult::Buffer_Error;
    auto *surface = reinterpret_cast<NvBufSurface *>(map.data);
    NvDsBatchMeta *batch = gst_buffer_get_nvds_batch_meta(buffer);
    if (!surface || !batch) {
      gst_buffer_unmap(buffer, &map);
      GST_ERROR("weed-bbox-meta: missing NvBufSurface or NvDsBatchMeta");
      return BufferResult::Buffer_Error;
    }

    cudaError_t cuda_status = cudaSetDevice(surface->gpuId);
    for (NvDsMetaList *node = batch->frame_meta_list; node && cuda_status == cudaSuccess;
         node = node->next) {
      auto *frame = static_cast<NvDsFrameMeta *>(node->data);
      const guint index = frame->batch_id;
      if (index >= surface->numFilled) continue;
      auto &plane = surface->surfaceList[index];
      if (plane.colorFormat != NVBUF_COLOR_FORMAT_RGBA || !plane.dataPtr) continue;
      const int width = static_cast<int>(plane.width);
      const int height = static_cast<int>(plane.height);
      const size_t mask_size = static_cast<size_t>(width) * height;
      if (mask_size != device_mask_size_) {
        if (device_mask_) cudaFree(device_mask_);
        device_mask_ = nullptr;
        cuda_status = cudaMalloc(reinterpret_cast<void **>(&device_mask_), mask_size);
        if (cuda_status != cudaSuccess) break;
        device_mask_size_ = mask_size;
      }

      const dim3 block(32, 8);
      const dim3 grid((width + block.x - 1) / block.x,
                      (height + block.y - 1) / block.y);
      make_exg_mask<<<grid, block>>>(static_cast<const uchar4 *>(plane.dataPtr),
                                    plane.pitch, device_mask_, width, height,
                                    threshold_);
      cuda_status = cudaGetLastError();
      if (cuda_status != cudaSuccess) break;

      std::vector<uint8_t> host_mask(mask_size);
      cuda_status = cudaMemcpy(host_mask.data(), device_mask_, mask_size,
                               cudaMemcpyDeviceToHost);
      if (cuda_status != cudaSuccess) break;

      cv::Mat mask(height, width, CV_8UC1, host_mask.data());
      // Exclude the convex hull of the YOLO chilli boxes, matching the prior
      // plant-hull suppression used by the Python postprocessor.
      std::vector<cv::Point> plant_corners;
      for (NvDsMetaList *obj_node = frame->obj_meta_list; obj_node;
           obj_node = obj_node->next) {
        auto *object = static_cast<NvDsObjectMeta *>(obj_node->data);
        if (object->class_id != 0) continue;
        const int x1 = std::clamp(static_cast<int>(object->rect_params.left), 0, width);
        const int y1 = std::clamp(static_cast<int>(object->rect_params.top), 0, height);
        const int x2 = std::clamp(static_cast<int>(object->rect_params.left + object->rect_params.width), 0, width);
        const int y2 = std::clamp(static_cast<int>(object->rect_params.top + object->rect_params.height), 0, height);
        if (x2 > x1 && y2 > y1) {
          plant_corners.emplace_back(x1, y1);
          plant_corners.emplace_back(x2 - 1, y1);
          plant_corners.emplace_back(x2 - 1, y2 - 1);
          plant_corners.emplace_back(x1, y2 - 1);
        }
      }
      if (!plant_corners.empty()) {
        std::vector<cv::Point> plant_hull;
        cv::convexHull(plant_corners, plant_hull);
        cv::fillConvexPoly(mask, plant_hull, cv::Scalar(0));
      }

      cv::Mat labels, stats, centroids;
      const int count = cv::connectedComponentsWithStats(mask, labels, stats, centroids, 8, CV_32S);
      std::vector<cv::Rect> boxes;
      boxes.reserve(std::min(count - 1, kMaxWeedsPerFrame));
      for (int label = 1; label < count; ++label) {
        const int area = stats.at<int>(label, cv::CC_STAT_AREA);
        const int x = stats.at<int>(label, cv::CC_STAT_LEFT);
        const int y = stats.at<int>(label, cv::CC_STAT_TOP);
        const int w = stats.at<int>(label, cv::CC_STAT_WIDTH);
        const int h = stats.at<int>(label, cv::CC_STAT_HEIGHT);
        if (area < min_area_) continue;
        const size_t first_new_box = boxes.size();
        add_component_boxes(labels, label, cv::Rect(x, y, w, h), area,
                            min_area_, cluster_area_, separation_px_, boxes);
        if (max_box_area_ > 0) {
          boxes.erase(std::remove_if(boxes.begin() + first_new_box, boxes.end(),
                                     [this](const cv::Rect &box) {
                                       return box.area() > max_box_area_;
                                     }),
                      boxes.end());
        }
      }
      std::sort(boxes.begin(), boxes.end(), [](const cv::Rect &a, const cv::Rect &b) {
        return a.area() > b.area();
      });
      if (boxes.size() > kMaxWeedsPerFrame) boxes.resize(kMaxWeedsPerFrame);

      nvds_acquire_meta_lock(batch);
      for (const auto &box : boxes) {
        NvDsObjectMeta *object = nvds_acquire_obj_meta_from_pool(batch);
        if (!object) continue;
        object->unique_component_id = 0;
        object->class_id = kWeedClassId;
        object->confidence = 1.0f;
        object->object_id = UNTRACKED_OBJECT_ID;
        g_strlcpy(object->obj_label, "weed", MAX_LABEL_SIZE);
        object->rect_params.left = static_cast<float>(box.x);
        object->rect_params.top = static_cast<float>(box.y);
        object->rect_params.width = static_cast<float>(box.width);
        object->rect_params.height = static_cast<float>(box.height);
        object->detector_bbox_info.org_bbox_coords.left = static_cast<float>(box.x);
        object->detector_bbox_info.org_bbox_coords.top = static_cast<float>(box.y);
        object->detector_bbox_info.org_bbox_coords.width = static_cast<float>(box.width);
        object->detector_bbox_info.org_bbox_coords.height = static_cast<float>(box.height);
        object->rect_params.border_width = 2;
        object->rect_params.border_color = {1.0, 0.0, 0.0, 1.0};
        nvds_add_obj_meta_to_frame(frame, object, nullptr);
      }
      nvds_release_meta_lock(batch);
    }
    gst_buffer_unmap(buffer, &map);
    if (cuda_status != cudaSuccess) {
      GST_ERROR("weed-bbox-meta CUDA failure: %s", cudaGetErrorString(cuda_status));
      return BufferResult::Buffer_Error;
    }
    return BufferResult::Buffer_Ok;
  }

 private:
  int threshold_ = 20;
  int min_area_ = kDefaultMinArea;
  int max_box_area_ = kDefaultMaxBoxArea;
  int cluster_area_ = 30000;
  int separation_px_ = 9;
  uint8_t *device_mask_ = nullptr;
  size_t device_mask_size_ = 0;
};
}  // namespace

extern "C" IDSCustomLibrary *CreateCustomAlgoCtx(DSCustom_CreateParams *) {
  return new WeedBboxMeta();
}
