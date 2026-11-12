DS_ROOT ?= /opt/nvidia/deepstream/deepstream
TARGET ?= libweed_bbox_meta.so
NVCC ?= nvcc
PKG_CONFIG ?= pkg-config

DS_INCLUDES = -I$(DS_ROOT)/sources/includes \
	-I$(DS_ROOT)/sources/gst-plugins/gst-nvdsvideotemplate/includes
DS_LIBS = -L$(DS_ROOT)/lib
NVCCFLAGS += -O3 -std=c++17 -Xcompiler=-fPIC,-pthread $(DS_INCLUDES) \
	$(shell $(PKG_CONFIG) --cflags gstreamer-1.0 gstreamer-base-1.0 gstreamer-video-1.0 opencv4 | sed 's/-pthread//g')
LDLIBS += $(DS_LIBS) $(shell $(PKG_CONFIG) --libs gstreamer-1.0 gstreamer-base-1.0 gstreamer-video-1.0 opencv4) \
	-lnvdsgst_helper -lnvdsgst_meta -lnvds_meta -lnvbufsurface \
	-lnvbufsurftransform -lnvdsbufferpool -Xlinker -rpath -Xlinker $(DS_ROOT)/lib

all: $(TARGET)

$(TARGET): weed_bbox_meta.cu
	$(NVCC) $(NVCCFLAGS) -shared -o $@ $< $(LDLIBS)

clean:
	rm -f $(TARGET)

.PHONY: all clean
