use std::ops::{Add, Div, Mul, Sub};

mod private {
    pub trait IntoMask<M> {
        fn into_mask(self) -> M;
    }
}

macro_rules! impl_math_op {
    ($trait:ident, $fn_name:ident, $type:ident, $intrinsic:ident) => {
        impl $trait for $type {
            type Output = Self;

            fn $fn_name(self, rhs: Self) -> Self::Output {
                unsafe { Self($intrinsic(self.0, rhs.0)) }
            }
        }
    };
}

macro_rules! impl_into_mask {
    ($raw:ty => $mask:ident) => {
        impl crate::math::simd_old::private::IntoMask<$mask> for $raw {
            fn into_mask(self) -> $mask {
                unsafe { $mask(std::mem::transmute(self)) }
            }
        }
    };
}


#[cfg(target_arch = "x86_64")]
pub mod arch {
    use super::*;
    use std::arch::x86_64::*;

    pub struct F32x4(pub(crate) __m128);
    pub struct F32x8(pub(crate) __m256);
    pub struct F64x2(pub(crate) __m128d);
    pub struct F64x4(pub(crate) __m256d);

    pub struct I8x16(pub(crate) __m128i);
    pub struct I8x32(pub(crate) __m256i);
    pub struct I16x8(pub(crate) __m128i);
    pub struct I16x16(pub(crate) __m256i);
    pub struct I32x4(pub(crate) __m128i);
    pub struct I32x8(pub(crate) __m256i);
    pub struct I64x2(pub(crate) __m128i);
    pub struct I64x4(pub(crate) __m256i);

    pub struct U8x16(pub(crate) __m128i);
    pub struct U8x32(pub(crate) __m256i);
    pub struct U16x8(pub(crate) __m128i);
    pub struct U16x16(pub(crate) __m256i);
    pub struct U32x4(pub(crate) __m128i);
    pub struct U32x8(pub(crate) __m256i);
    pub struct U64x2(pub(crate) __m128i);
    pub struct U64x4(pub(crate) __m256i);


    pub struct Mask8x16(pub(crate) __m128);
    pub struct Mask8x32(pub(crate) __m256);
    pub struct Mask16x8(pub(crate) __m128);
    pub struct Mask16x16(pub(crate) __m256);
    pub struct Mask32x4(pub(crate) __m128);
    pub struct Mask32x8(pub(crate) __m256);
    pub struct Mask64x2(pub(crate) __m128);
    pub struct Mask64x4(pub(crate) __m256);


    impl_math_op!(Add, add, F32x4, _mm_add_ps);
    impl_math_op!(Sub, sub, F32x4, _mm_sub_ps);
    impl_math_op!(Mul, mul, F32x4, _mm_mul_ps);
    impl_math_op!(Div, div, F32x4, _mm_div_ps);

    impl_math_op!(Add, add, F32x8, _mm256_add_ps);
    impl_math_op!(Sub, sub, F32x8, _mm256_sub_ps);
    impl_math_op!(Mul, mul, F32x8, _mm256_mul_ps);
    impl_math_op!(Div, div, F32x8, _mm256_div_ps);

    impl_math_op!(Add, add, F64x4, _mm256_add_pd);
    impl_math_op!(Sub, sub, F64x4, _mm256_sub_pd);
    impl_math_op!(Mul, mul, F64x4, _mm256_mul_pd);
    impl_math_op!(Div, div, F64x4, _mm256_div_pd);

    // F64x2
    impl_math_op!(Add, add, F64x2, _mm_add_pd);
    impl_math_op!(Sub, sub, F64x2, _mm_sub_pd);
    impl_math_op!(Mul, mul, F64x2, _mm_mul_pd);
    impl_math_op!(Div, div, F64x2, _mm_div_pd);

    // I8x16
    impl_math_op!(Add, add, I8x16, _mm_add_epi8);
    impl_math_op!(Sub, sub, I8x16, _mm_sub_epi8);
    // no mul, no div for i8

    // I8x32
    impl_math_op!(Add, add, I8x32, _mm256_add_epi8);
    impl_math_op!(Sub, sub, I8x32, _mm256_sub_epi8);
    // no mul, no div for i8

    // I16x8
    impl_math_op!(Add, add, I16x8, _mm_add_epi16);
    impl_math_op!(Sub, sub, I16x8, _mm_sub_epi16);
    impl_math_op!(Mul, mul, I16x8, _mm_mullo_epi16);
    // no div for i16

    // I16x16
    impl_math_op!(Add, add, I16x16, _mm256_add_epi16);
    impl_math_op!(Sub, sub, I16x16, _mm256_sub_epi16);
    impl_math_op!(Mul, mul, I16x16, _mm256_mullo_epi16);
    // no div for i16

    // I32x4
    impl_math_op!(Add, add, I32x4, _mm_add_epi32);
    impl_math_op!(Sub, sub, I32x4, _mm_sub_epi32);
    impl_math_op!(Mul, mul, I32x4, _mm_mullo_epi32);
    // no div for i32

    // I32x8
    impl_math_op!(Add, add, I32x8, _mm256_add_epi32);
    impl_math_op!(Sub, sub, I32x8, _mm256_sub_epi32);
    impl_math_op!(Mul, mul, I32x8, _mm256_mullo_epi32);
    // no div for i32

    // I64x2
    impl_math_op!(Add, add, I64x2, _mm_add_epi64);
    impl_math_op!(Sub, sub, I64x2, _mm_sub_epi64);
    // no mul, no div for i64 on AVX2

    // I64x4
    impl_math_op!(Add, add, I64x4, _mm256_add_epi64);
    impl_math_op!(Sub, sub, I64x4, _mm256_sub_epi64);
    // no mul, no div for i64 on AVX2

    // U8x16 — unsigned uses same epi intrinsics for add/sub (wrapping is wrapping)
    impl_math_op!(Add, add, U8x16, _mm_add_epi8);
    impl_math_op!(Sub, sub, U8x16, _mm_sub_epi8);
    // no mul, no div for u8

    // U8x32
    impl_math_op!(Add, add, U8x32, _mm256_add_epi8);
    impl_math_op!(Sub, sub, U8x32, _mm256_sub_epi8);
    // no mul, no div for u8

    // U16x8
    impl_math_op!(Add, add, U16x8, _mm_add_epi16);
    impl_math_op!(Sub, sub, U16x8, _mm_sub_epi16);
    impl_math_op!(Mul, mul, U16x8, _mm_mullo_epi16);
    // no div for u16

    // U16x16
    impl_math_op!(Add, add, U16x16, _mm256_add_epi16);
    impl_math_op!(Sub, sub, U16x16, _mm256_sub_epi16);
    impl_math_op!(Mul, mul, U16x16, _mm256_mullo_epi16);
    // no div for u16

    // U32x4
    impl_math_op!(Add, add, U32x4, _mm_add_epi32);
    impl_math_op!(Sub, sub, U32x4, _mm_sub_epi32);
    impl_math_op!(Mul, mul, U32x4, _mm_mullo_epi32);
    // no div for u32

    // U32x8
    impl_math_op!(Add, add, U32x8, _mm256_add_epi32);
    impl_math_op!(Sub, sub, U32x8, _mm256_sub_epi32);
    impl_math_op!(Mul, mul, U32x8, _mm256_mullo_epi32);
    // no div for u32

    // U64x2
    impl_math_op!(Add, add, U64x2, _mm_add_epi64);
    impl_math_op!(Sub, sub, U64x2, _mm_sub_epi64);
    // no mul, no div for u64 on AVX2

    // U64x4
    impl_math_op!(Add, add, U64x4, _mm256_add_epi64);
    impl_math_op!(Sub, sub, U64x4, _mm256_sub_epi64);
    // no mul, no div for u64 on AVX2

    // 128-bit
    impl_into_mask!(__m128  => Mask8x16);
    impl_into_mask!(__m128i => Mask8x16);
    impl_into_mask!(__m128  => Mask16x8);
    impl_into_mask!(__m128i => Mask16x8);
    impl_into_mask!(__m128  => Mask32x4);
    impl_into_mask!(__m128i => Mask32x4);
    impl_into_mask!(__m128d => Mask32x4);
    impl_into_mask!(__m128  => Mask64x2);
    impl_into_mask!(__m128i => Mask64x2);
    impl_into_mask!(__m128d => Mask64x2);

    // 256-bit
    impl_into_mask!(__m256  => Mask8x32);
    impl_into_mask!(__m256i => Mask8x32);
    impl_into_mask!(__m256  => Mask16x16);
    impl_into_mask!(__m256i => Mask16x16);
    impl_into_mask!(__m256  => Mask32x8);
    impl_into_mask!(__m256i => Mask32x8);
    impl_into_mask!(__m256d => Mask32x8);
    impl_into_mask!(__m256  => Mask64x4);
    impl_into_mask!(__m256i => Mask64x4);
    impl_into_mask!(__m256d => Mask64x4);
}

#[cfg(target_arch = "aarch64")]
pub mod arch {}
