// https://raw.githubusercontent.com/Tencent/ncnn/master/src/layer/x86/avx_mathfun.h

#include <cstdint>
#ifdef __SIMD__
#include <mipp/mipp.h>
#endif

const struct {
  float LowerRange;
  float UpperRange;
  float alpha_13;
  float alpha_11;
  float alpha_9;
  float alpha_7;
  float alpha_5;
  float alpha_3;
  float alpha_1;
  float beta_6;
  float beta_4;
  float beta_2;
  float beta_0;
} MlasTanhConstants = {
    -9.0f,
    9.0f,
    -2.76076847742355e-16f,
    2.00018790482477e-13f,
    -8.60467152213735e-11f,
    5.12229709037114e-08f,
    1.48572235717979e-05f,
    6.37261928875436e-04f,
    4.89352455891786e-03f,
    1.19825839466702e-06f,
    1.18534705686654e-04f,
    2.26843463243900e-03f,
    4.89352518554385e-03f,
};

template <class T, class... Rest>
constexpr bool is_any = (std::is_same_v<T, Rest> || ...);

template <typename VEC_T> VEC_T tanh_mlas(VEC_T Value) {
  // This odd two-step process exists to ensure an input value of NaN carries
  // through without modification because "std::min" and "std::max" return
  // unreliable results when NaNs are involved, and it's clear from the test's
  // reference outputs that they want a NaN on output whenever the input is a
  // NaN.
  //if constexpr (std::is_same_v<VEC_T, float>) {
  if constexpr (is_any<VEC_T, float, int32_t, uint32_t, double, int64_t,
                       uint64_t>) {
    auto v_tmp = (Value < MlasTanhConstants.LowerRange)? MlasTanhConstants.LowerRange: Value;
    Value = (v_tmp > MlasTanhConstants.UpperRange) ? MlasTanhConstants.UpperRange: v_tmp;
  } else {
#ifdef __SIMD__
    auto lower_range = VEC_T(MlasTanhConstants.LowerRange);
    auto upper_range = VEC_T(MlasTanhConstants.UpperRange);
    auto v_tmp = mipp::min(Value, upper_range);
    Value = mipp::max(v_tmp, lower_range);
#else
    abort();
  #endif
  }

  auto ValueSquared = Value * Value;

  auto p =
      ValueSquared * MlasTanhConstants.alpha_13 + MlasTanhConstants.alpha_11;
  p = p * ValueSquared + MlasTanhConstants.alpha_9;
  p = p * ValueSquared + MlasTanhConstants.alpha_7;
  p = p * ValueSquared + MlasTanhConstants.alpha_5;
  p = p * ValueSquared + MlasTanhConstants.alpha_3;
  p = p * ValueSquared + MlasTanhConstants.alpha_1;
  p = p * Value;

  auto q = ValueSquared * MlasTanhConstants.beta_6 + MlasTanhConstants.beta_4;
  q = q * ValueSquared + MlasTanhConstants.beta_2;
  q = q * ValueSquared + MlasTanhConstants.beta_0;

  return (p / q);
}