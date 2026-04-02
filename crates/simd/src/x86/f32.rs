use crate::*;
use std::arch::x86_64::*;

pub struct F32x4(pub(crate) __m128);

impl F32x4 {
    impl_init!(
        f32,
        4,
        _mm_setzero_ps,
        _mm_set1_ps,
        _mm_loadu_ps,
        _mm_load_ps,
        _mm_storeu_ps,
        _mm_store_ps
    );

    impl_intr!(add(self, rhs) => _mm_add_ps);
    impl_intr!(sub(self, rhs) => _mm_sub_ps);
    impl_intr!(mul(self, rhs) => _mm_mul_ps);
    impl_intr!(div(self, rhs) => _mm_div_ps);
    impl_intr!(fmadd(self, b, c) => _mm_fmadd_ps);
    impl_intr!(fnmadd(self, b, c) => _mm_fnmadd_ps);
    impl_intr!(sqrt(self) => _mm_sqrt_ps);
    impl_intr!(min(self, rhs) => _mm_min_ps);
    impl_intr!(max(self, rhs) => _mm_max_ps);
    impl_intr!(ceil(self) => _mm_ceil_ps);
    impl_intr!(floor(self) => _mm_floor_ps);
    impl_intr!(round(self) => _mm_round_ps::<_MM_FROUND_TO_NEAREST_INT>);
    impl_intr!(round_up(self) => _mm_round_ps::<_MM_FROUND_TO_POS_INF>);
    impl_intr!(round_down(self) => _mm_round_ps::<_MM_FROUND_TO_NEG_INF>);
    impl_intr!(round_to_zero(self) => _mm_round_ps::<_MM_FROUND_TO_ZERO>);
    impl_intr!(bitand(self, rhs) => _mm_and_ps);
    impl_intr!(bitor(self, rhs) => _mm_or_ps);
    impl_intr!(bitxor(self, rhs) => _mm_xor_ps);
    impl_intr!(bitandnot(self, rhs) => _mm_andnot_ps);

    // abs

    pub fn neg(self) -> Self {
        let sign_mask = F32x4::splat(-0.0);
        unsafe { Self(_mm_xor_ps(self.0, sign_mask.0)) }
    }
}
