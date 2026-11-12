# Build on the same Jetson that runs DeepStream; do not copy this .so across SDK releases.
DS_PATH ?= /opt/nvidia/deepstream/deepstream
CUDA_PATH ?= /usr/local/cuda
TARGET := ../libweed_mask_template.so
NVCC := $(CUDA_PATH)/bin/nvcc
CXXFLAGS := -O3 -std=c++17 -Xcompiler -fPIC -Xcompiler -pthread -shared
GST_CFLAGS := $(filter-out -pthread,$(shell pkg-config --cflags gstreamer-1.0 gstreamer-video-1.0))
INCLUDES := -I$(DS_PATH)/sources/includes -I$(DS_PATH)/sources/gst-plugins/gst-nvdsvideotemplate/includes \
            -I$(CUDA_PATH)/include $(GST_CFLAGS)
LIBS := -L$(CUDA_PATH)/lib64 -lcudart -ldl -lpthread \
        -L$(DS_PATH)/lib -lnvdsgst_helper -lnvdsgst_meta -lnvds_meta \
        -lnvbufsurface -lnvbufsurftransform -lnvdsbufferpool \
        -Xlinker -rpath -Xlinker $(DS_PATH)/lib $(shell pkg-config --libs gstreamer-1.0 gstreamer-video-1.0)

all: $(TARGET)
$(TARGET): weed_mask_template.cu
	$(NVCC) $(CXXFLAGS) $(INCLUDES) $< -o $@ $(LIBS)
clean:
	rm -f $(TARGET)
