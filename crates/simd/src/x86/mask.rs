use crate::*;
use std::arch::x86_64::*;

pub(crate) trait IntoMask<M> {
    fn into_mask(self) -> M;
}

#[macro_export]
macro_rules! impl_into_mask {
    ($raw:ty => $mask:ident) => {
        impl IntoMask<$mask> for $raw {
            fn into_mask(self) -> $mask {
                unsafe { $mask(std::mem::transmute(self)) }
            }
        }
    };
}

pub struct Mask32x4(pub(crate) __m128);

impl Mask32x4 {
    pub fn splat(b: bool) -> Self {
        let mask = (b as i32).wrapping_neg();
        unsafe { Self(_mm_castsi128_ps(_mm_set1_epi32(mask))) }
    }

    pub fn from_array(b: [bool; 4]) -> Self {
        let masks = [
            (b[0] as i32).wrapping_neg(),
            (b[1] as i32).wrapping_neg(),
            (b[2] as i32).wrapping_neg(),
            (b[3] as i32).wrapping_neg(),
        ];

        unsafe { Self(_mm_castsi128_ps(_mm_loadu_epi32(masks.as_ptr()))) }
    }

    pub fn bitmask(self) -> u32 {
        unsafe { _mm_movemask_ps(self.0) as u32 }
    }

    pub fn any(self) -> bool {
        self.bitmask() != 0
    }

    pub fn all_true(self) -> bool {
        self.bitmask() == (1u32 << 4) - 1
    }

    pub fn none(self) -> bool {
        self.bitmask() == 0
    }

    impl_intr!(and(self, rhs) => _mm_and_ps);
    impl_intr!(or(self, rhs) => _mm_or_ps);
    impl_intr!(xor(self, rhs) => _mm_xor_ps);
    impl_intr!(andnot(self, rhs) => _mm_andnot_ps);

    pub fn not(self) -> Self {
        Self::splat(true).andnot(self)
    }
}
