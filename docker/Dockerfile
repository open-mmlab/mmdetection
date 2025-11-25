ARG CU_VERSION="129"
ARG TORCH_VERSION="280"
ARG ONEDL_MMPRETRAIN_VERSION="1.3.1.post1"

FROM vbti/onedl-mmpretrain-cu${CU_VERSION}-torch${TORCH_VERSION}:${ONEDL_MMPRETRAIN_VERSION}

# Install OneDL-MMDetection
COPY . /onedl/onedl-mmdetection

WORKDIR /onedl/onedl-mmdetection
RUN mim install --no-cache-dir -e .[multimodal,optional,albu,tracking] \
    && python -m compileall -q /onedl/onedl-mmdetection

# Verify installation
RUN python -c "import mmdet; print('mmdet version:', mmdet.__version__)"
