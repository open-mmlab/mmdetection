__device__ inline bool devIoU(float const *const a, float const *const b,
                              const int offset, const float threshold) {
  float left = fmaxf(a[0], b[0]), right = fminf(a[2], b[2]);
  float top = fmaxf(a[1], b[1]), bottom = fminf(a[3], b[3]);
  float width = fmaxf(right - left + offset, 0.f),
        height = fmaxf(bottom - top + offset, 0.f);
  float interS = width * height;
  float Sa = (a[2] - a[0] + offset) * (a[3] - a[1] + offset);
  float Sb = (b[2] - b[0] + offset) * (b[3] - b[1] + offset);
  return interS > threshold * (Sa + Sb - interS);
}

__device__ inline bool devIoU(float const *const a, float const *const b,
                              const int offset, const float threshold) {
  float left = fmaxf(a[0], b[0]), right = fminf(a[2], b[2]);
  float top = fmaxf(a[1], b[1]), bottom = fminf(a[3], b[3]);
  float width = fmaxf(right - left + offset, 0.f),
        height = fmaxf(bottom - top + offset, 0.f);
  float interS = width * height;
  float Sa = (a[2] - a[0] + offset) * (a[3] - a[1] + offset);
  float Sb = (b[2] - b[0] + offset) * (b[3] - b[1] + offset);
  
  // calculate enclosing box coords for DIoU-NMS
  float eleft = fminf(a[0], b[0]); eright = fmaxf(a[2], b[2]);
  float etop = fminf(a[1], b[1]), ebottom = fmaxf(a[3], b[3]);

  // calculate centers coords for DIoU-NMS
  float xcentera = (a[0] + a[2]) / 2; xcenterb = (b[0] + b[2]) / 2;
  float ycentera = (a[1] + a[3]) / 2, ycenterb = (b[1] + b[3]) / 2;

  float diagonal = ((eright - eleft) ** 2) + ((ebottom - etop) ** 2) + 1e-7
  float distance = ((xcentera - xcenterb) ** 2) + ((ycentera - ycenterb) ** 2)
  float penalty = distance / diagonal

  return interS > threshold * (Sa + Sb - interS - penalty);
}