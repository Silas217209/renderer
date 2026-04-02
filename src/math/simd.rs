use std::ops::{Add, BitAnd, BitOr, BitXor, Div, Mul, Not, Sub};

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
        impl crate::math::simd::private::IntoMask<$mask> for $raw {
            fn into_mask(self) -> $mask {
                unsafe { $mask(std::mem::transmute(self)) }
            }
        }
    };
}

macro_rules! impl_init {
    (
        $type:ident,        // F32x4
        $scalar:ty,         // f32
        $lanes:literal,     // 4
        $zero:ident,        // _mm_setzero_ps
        $splat:ident,       // _mm_set1_ps
        $load:ident         // _mm_loadu_ps
        $(, $param:ident)*  // a, b, c, d  — variadic lane params for new()
    ) => {
        impl $type {
            pub fn zero() -> Self {
                unsafe { Self($zero()) }
            }
            pub fn splat(scalar: $scalar) -> Self {
                unsafe { Self($splat(scalar.try_into().unwrap())) }
            }
            pub fn from_slice(data: &[$scalar]) -> Self {
                assert!(data.len() >= $lanes);
                unsafe { Self($load(data.as_ptr() as *const _)) }
            }
            pub fn from_array(data: [$scalar; $lanes]) -> Self {
                unsafe { Self($load(data.as_ptr() as *const _)) }
            }
        }
    };
}

macro_rules! impl_mask_init {
    ($type:ident, $zero:ident, $cast:ident, $cmpeq:ident) => {
        impl $type {
            pub fn zero() -> Self {
                unsafe { Self($cast($zero())) }
            }
            pub fn all() -> Self {
                unsafe {
                    let z = $zero();
                    Self($cast($cmpeq(z, z)))
                }
            }
        }
    };
}

macro_rules! impl_mask_ops {
    ($type:ident, $and:ident, $or:ident, $xor:ident, $andnot:ident) => {
        impl BitAnd for $type {
            type Output = Self;
            fn bitand(self, rhs: Self) -> Self {
                unsafe { Self($and(self.0, rhs.0)) }
            }
        }
        impl BitOr for $type {
            type Output = Self;
            fn bitor(self, rhs: Self) -> Self {
                unsafe { Self($or(self.0, rhs.0)) }
            }
        }
        impl BitXor for $type {
            type Output = Self;
            fn bitxor(self, rhs: Self) -> Self {
                unsafe { Self($xor(self.0, rhs.0)) }
            }
        }
        impl Not for $type {
            type Output = Self;
            fn not(self) -> Self {
                // andnot(a, b) = !a & b, so andnot(self, all) = !self
                unsafe { Self($andnot(self.0, $type::all().0)) }
            }
        }
    };
}

macro_rules! impl_cmp {
    ($type:ident, $mask:ident, $eq:tt, $lt:tt, $gt:tt, $le:tt, $ge:tt, $ne:tt) => {
        impl $type {
            pub fn eq(self, rhs: Self) -> $mask {
                unsafe { ($eq)(self.0, rhs.0).into_mask() }
            }
            pub fn ne(self, rhs: Self) -> $mask {
                unsafe { ($ne)(self.0, rhs.0).into_mask() }
            }
            pub fn lt(self, rhs: Self) -> $mask {
                unsafe { ($lt)(self.0, rhs.0).into_mask() }
            }
            pub fn gt(self, rhs: Self) -> $mask {
                unsafe { ($gt)(self.0, rhs.0).into_mask() }
            }
            pub fn le(self, rhs: Self) -> $mask {
                unsafe { ($le)(self.0, rhs.0).into_mask() }
            }
            pub fn ge(self, rhs: Self) -> $mask {
                unsafe { ($ge)(self.0, rhs.0).into_mask() }
            }
        }
    };
}

#[cfg(target_arch = "x86_64")]
pub mod arch {
    use super::*;
    use crate::math::simd::private::IntoMask;
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

    #[rustfmt::skip]
    impl_init!(F32x4, f32, 4, _mm_setzero_ps, _mm_set1_ps, _mm_loadu_ps);
    #[rustfmt::skip]
    impl_init!(F32x8, f32, 8, _mm256_setzero_ps, _mm256_set1_ps, _mm256_loadu_ps);
    #[rustfmt::skip]
    impl_init!(F64x2, f64, 4, _mm_setzero_pd, _mm_set1_pd, _mm_loadu_pd);
    #[rustfmt::skip]
    impl_init!(F64x4, f64, 8, _mm256_setzero_pd, _mm256_set1_pd, _mm256_loadu_pd);

    #[rustfmt::skip]
    impl_init!(I8x16, i8, 16, _mm_setzero_si128, _mm_set1_epi8, _mm_loadu_epi8);
    #[rustfmt::skip]
    impl_init!(I8x32, i8, 32, _mm256_setzero_si256, _mm256_set1_epi8, _mm256_loadu_epi8);
    #[rustfmt::skip]
    impl_init!(I16x8, i16, 8, _mm_setzero_si128, _mm_set1_epi16, _mm_loadu_epi16);
    #[rustfmt::skip]
    impl_init!(I16x16, i16, 16, _mm256_setzero_si256, _mm256_set1_epi16, _mm256_loadu_epi16);
    #[rustfmt::skip]
    impl_init!(I32x4, i32, 4, _mm_setzero_si128, _mm_set1_epi32, _mm_loadu_epi32);
    #[rustfmt::skip]
    impl_init!(I32x8, i32, 8, _mm256_setzero_si256, _mm256_set1_epi32, _mm256_loadu_epi32);
    #[rustfmt::skip]
    impl_init!(I64x2, i64, 2, _mm_setzero_si128, _mm_set1_epi64x, _mm_loadu_epi64);
    #[rustfmt::skip]
    impl_init!(I64x4, i64, 4, _mm256_setzero_si256, _mm256_set1_epi64x, _mm256_loadu_epi64);

    #[rustfmt::skip]
    impl_init!(U8x16, u8, 16, _mm_setzero_si128, _mm_set1_epi8, _mm_loadu_epi8);
    #[rustfmt::skip]
    impl_init!(U8x32, u8, 32, _mm256_setzero_si256, _mm256_set1_epi8, _mm256_loadu_epi8);
    #[rustfmt::skip]
    impl_init!(U16x8, u16, 8, _mm_setzero_si128, _mm_set1_epi16, _mm_loadu_epi16);
    #[rustfmt::skip]
    impl_init!(U16x16, u16, 16, _mm256_setzero_si256, _mm256_set1_epi16, _mm256_loadu_epi16);
    #[rustfmt::skip]
    impl_init!(U32x4, u32, 4, _mm_setzero_si128, _mm_set1_epi32, _mm_loadu_epi32);
    #[rustfmt::skip]
    impl_init!(U32x8, u32, 8, _mm256_setzero_si256, _mm256_set1_epi32, _mm256_loadu_epi32);
    #[rustfmt::skip]
    impl_init!(U64x2, u64, 2, _mm_setzero_si128, _mm_set1_epi64x, _mm_loadu_epi64);
    #[rustfmt::skip]
    impl_init!(U64x4, u64, 4, _mm256_setzero_si256, _mm256_set1_epi64x, _mm256_loadu_epi64);

    #[rustfmt::skip]
    impl_mask_init!(Mask8x16,  _mm_setzero_si128,    _mm_castsi128_ps,    _mm_cmpeq_epi8);
    #[rustfmt::skip]
    impl_mask_init!(Mask8x32,  _mm256_setzero_si256, _mm256_castsi256_ps, _mm256_cmpeq_epi8);
    #[rustfmt::skip]
    impl_mask_init!(Mask16x8,  _mm_setzero_si128,    _mm_castsi128_ps,    _mm_cmpeq_epi16);
    #[rustfmt::skip]
    impl_mask_init!(Mask16x16, _mm256_setzero_si256, _mm256_castsi256_ps, _mm256_cmpeq_epi16);
    #[rustfmt::skip]
    impl_mask_init!(Mask32x4,  _mm_setzero_si128,    _mm_castsi128_ps,    _mm_cmpeq_epi32);
    #[rustfmt::skip]
    impl_mask_init!(Mask32x8,  _mm256_setzero_si256, _mm256_castsi256_ps, _mm256_cmpeq_epi32);
    #[rustfmt::skip]
    impl_mask_init!(Mask64x2,  _mm_setzero_si128,    _mm_castsi128_ps,    _mm_cmpeq_epi64);
    #[rustfmt::skip]
    impl_mask_init!(Mask64x4,  _mm256_setzero_si256, _mm256_castsi256_ps, _mm256_cmpeq_epi64);

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

    #[rustfmt::skip]
    impl_mask_ops!(Mask8x16,  _mm_and_ps,    _mm_or_ps,    _mm_xor_ps,    _mm_andnot_ps);
    #[rustfmt::skip]
    impl_mask_ops!(Mask8x32,  _mm256_and_ps, _mm256_or_ps, _mm256_xor_ps, _mm256_andnot_ps);
    #[rustfmt::skip]
    impl_mask_ops!(Mask16x8,  _mm_and_ps,    _mm_or_ps,    _mm_xor_ps,    _mm_andnot_ps);
    #[rustfmt::skip]
    impl_mask_ops!(Mask16x16, _mm256_and_ps, _mm256_or_ps, _mm256_xor_ps, _mm256_andnot_ps);
    #[rustfmt::skip]
    impl_mask_ops!(Mask32x4,  _mm_and_ps,    _mm_or_ps,    _mm_xor_ps,    _mm_andnot_ps);
    #[rustfmt::skip]
    impl_mask_ops!(Mask32x8,  _mm256_and_ps, _mm256_or_ps, _mm256_xor_ps, _mm256_andnot_ps);
    #[rustfmt::skip]
    impl_mask_ops!(Mask64x2,  _mm_and_ps,    _mm_or_ps,    _mm_xor_ps,    _mm_andnot_ps);
    #[rustfmt::skip]
    impl_mask_ops!(Mask64x4,  _mm256_and_ps, _mm256_or_ps, _mm256_xor_ps, _mm256_andnot_ps);

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

    #[rustfmt::skip]
    impl_cmp!(F32x4, Mask32x4,
        (|a, b| _mm_cmp_ps(a, b, _CMP_EQ_OQ)),
        (|a, b| _mm_cmp_ps(a, b, _CMP_LT_OQ)),
        (|a, b| _mm_cmp_ps(a, b, _CMP_GT_OQ)),
        (|a, b| _mm_cmp_ps(a, b, _CMP_LE_OQ)),
        (|a, b| _mm_cmp_ps(a, b, _CMP_GE_OQ)),
        (|a, b| _mm_cmp_ps(a, b, _CMP_NEQ_OQ))
    );
    #[rustfmt::skip]
    impl_cmp!(F32x8, Mask32x8,
        (|a, b| _mm256_cmp_ps(a, b, _CMP_EQ_OQ)),
        (|a, b| _mm256_cmp_ps(a, b, _CMP_LT_OQ)),
        (|a, b| _mm256_cmp_ps(a, b, _CMP_GT_OQ)),
        (|a, b| _mm256_cmp_ps(a, b, _CMP_LE_OQ)),
        (|a, b| _mm256_cmp_ps(a, b, _CMP_GE_OQ)),
        (|a, b| _mm256_cmp_ps(a, b, _CMP_NEQ_OQ))
    );
    #[rustfmt::skip]
    impl_cmp!(F64x2, Mask64x2,
        (|a, b| _mm_cmp_pd(a, b, _CMP_EQ_OQ)),
        (|a, b| _mm_cmp_pd(a, b, _CMP_LT_OQ)),
        (|a, b| _mm_cmp_pd(a, b, _CMP_GT_OQ)),
        (|a, b| _mm_cmp_pd(a, b, _CMP_LE_OQ)),
        (|a, b| _mm_cmp_pd(a, b, _CMP_GE_OQ)),
        (|a, b| _mm_cmp_pd(a, b, _CMP_NEQ_OQ))
    );
    #[rustfmt::skip]
    impl_cmp!(F64x4, Mask64x4,
        (|a, b| _mm256_cmp_pd(a, b, _CMP_EQ_OQ)),
        (|a, b| _mm256_cmp_pd(a, b, _CMP_LT_OQ)),
        (|a, b| _mm256_cmp_pd(a, b, _CMP_GT_OQ)),
        (|a, b| _mm256_cmp_pd(a, b, _CMP_LE_OQ)),
        (|a, b| _mm256_cmp_pd(a, b, _CMP_GE_OQ)),
        (|a, b| _mm256_cmp_pd(a, b, _CMP_NEQ_OQ))
    );
    #[rustfmt::skip]
    impl_cmp!(I8x16, Mask8x16,
        _mm_cmpeq_epi8,
        _mm_cmplt_epi8,
        _mm_cmpgt_epi8,
        (|a, b| _mm_or_si128(_mm_cmplt_epi8(a, b),  _mm_cmpeq_epi8(a, b))),
        (|a, b| _mm_or_si128(_mm_cmpgt_epi8(a, b),  _mm_cmpeq_epi8(a, b))),
        (|a, b| _mm_xor_si128(_mm_cmpeq_epi8(a, b), _mm_set1_epi8(-1)))
    );
    #[rustfmt::skip]
    impl_cmp!(I8x32, Mask8x32,
        _mm256_cmpeq_epi8,
        (|a, b| _mm256_cmpgt_epi8(b, a)),
        _mm256_cmpgt_epi8,
        (|a, b| _mm256_or_si256(_mm256_cmpgt_epi8(b, a), _mm256_cmpeq_epi8(a, b))),
        (|a, b| _mm256_or_si256(_mm256_cmpgt_epi8(a, b), _mm256_cmpeq_epi8(a, b))),
        (|a, b| _mm256_xor_si256(_mm256_cmpeq_epi8(a, b), _mm256_set1_epi8(-1)))
    );
    #[rustfmt::skip]
    impl_cmp!(I16x8, Mask16x8,
        _mm_cmpeq_epi16,
        _mm_cmplt_epi16,
        _mm_cmpgt_epi16,
        (|a, b| _mm_or_si128(_mm_cmplt_epi16(a, b), _mm_cmpeq_epi16(a, b))),
        (|a, b| _mm_or_si128(_mm_cmpgt_epi16(a, b), _mm_cmpeq_epi16(a, b))),
        (|a, b| _mm_xor_si128(_mm_cmpeq_epi16(a, b), _mm_set1_epi16(-1)))
    );
    #[rustfmt::skip]
    impl_cmp!(I16x16, Mask16x16,
        _mm256_cmpeq_epi16,
        (|a, b| _mm256_cmpgt_epi16(b, a)),
        _mm256_cmpgt_epi16,
        (|a, b| _mm256_or_si256(_mm256_cmpgt_epi16(b, a), _mm256_cmpeq_epi16(a, b))),
        (|a, b| _mm256_or_si256(_mm256_cmpgt_epi16(a, b), _mm256_cmpeq_epi16(a, b))),
        (|a, b| _mm256_xor_si256(_mm256_cmpeq_epi16(a, b), _mm256_set1_epi16(-1)))
    );
    #[rustfmt::skip]
    impl_cmp!(I32x4, Mask32x4,
        _mm_cmpeq_epi32,
        _mm_cmplt_epi32,
        _mm_cmpgt_epi32,
        (|a, b| _mm_or_si128(_mm_cmplt_epi32(a, b), _mm_cmpeq_epi32(a, b))),
        (|a, b| _mm_or_si128(_mm_cmpgt_epi32(a, b), _mm_cmpeq_epi32(a, b))),
        (|a, b| _mm_xor_si128(_mm_cmpeq_epi32(a, b), _mm_set1_epi32(-1)))
    );
    #[rustfmt::skip]
    impl_cmp!(I32x8, Mask32x8,
        _mm256_cmpeq_epi32,
        (|a, b| _mm256_cmpgt_epi32(b, a)),
        _mm256_cmpgt_epi32,
        (|a, b| _mm256_or_si256(_mm256_cmpgt_epi32(b, a), _mm256_cmpeq_epi32(a, b))),
        (|a, b| _mm256_or_si256(_mm256_cmpgt_epi32(a, b), _mm256_cmpeq_epi32(a, b))),
        (|a, b| _mm256_xor_si256(_mm256_cmpeq_epi32(a, b), _mm256_set1_epi32(-1)))
    );
    #[rustfmt::skip]
    impl_cmp!(I64x2, Mask64x2,
        _mm_cmpeq_epi64,
        (|a, b| _mm_cmpgt_epi64(b, a)),
        _mm_cmpgt_epi64,
        (|a, b| _mm_or_si128(_mm_cmpgt_epi64(b, a), _mm_cmpeq_epi64(a, b))),
        (|a, b| _mm_or_si128(_mm_cmpgt_epi64(a, b), _mm_cmpeq_epi64(a, b))),
        (|a, b| _mm_xor_si128(_mm_cmpeq_epi64(a, b), _mm_set1_epi64x(-1)))
    );
    #[rustfmt::skip]
    impl_cmp!(I64x4, Mask64x4,
        _mm256_cmpeq_epi64,
        (|a, b| _mm256_cmpgt_epi64(b, a)),
        _mm256_cmpgt_epi64,
        (|a, b| _mm256_or_si256(_mm256_cmpgt_epi64(b, a), _mm256_cmpeq_epi64(a, b))),
        (|a, b| _mm256_or_si256(_mm256_cmpgt_epi64(a, b), _mm256_cmpeq_epi64(a, b))),
        (|a, b| _mm256_xor_si256(_mm256_cmpeq_epi64(a, b), _mm256_set1_epi64x(-1)))
    );
    #[rustfmt::skip]
    impl_cmp!(U8x16, Mask8x16,
        _mm_cmpeq_epi8,
        (|a, b| { let bias = _mm_set1_epi8(i8::MIN); _mm_cmplt_epi8(_mm_xor_si128(a, bias), _mm_xor_si128(b, bias)) }),
        (|a, b| { let bias = _mm_set1_epi8(i8::MIN); _mm_cmpgt_epi8(_mm_xor_si128(a, bias), _mm_xor_si128(b, bias)) }),
        (|a, b| { let bias = _mm_set1_epi8(i8::MIN); _mm_or_si128(_mm_cmplt_epi8(_mm_xor_si128(a, bias), _mm_xor_si128(b, bias)), _mm_cmpeq_epi8(a, b)) }),
        (|a, b| { let bias = _mm_set1_epi8(i8::MIN); _mm_or_si128(_mm_cmpgt_epi8(_mm_xor_si128(a, bias), _mm_xor_si128(b, bias)), _mm_cmpeq_epi8(a, b)) }),
        (|a, b| _mm_xor_si128(_mm_cmpeq_epi8(a, b), _mm_set1_epi8(-1)))
    );
    #[rustfmt::skip]
    impl_cmp!(U8x32, Mask8x32,
        _mm256_cmpeq_epi8,
        (|a, b| { let bias = _mm256_set1_epi8(i8::MIN); _mm256_cmpgt_epi8(_mm256_xor_si256(b, bias), _mm256_xor_si256(a, bias)) }),
        (|a, b| { let bias = _mm256_set1_epi8(i8::MIN); _mm256_cmpgt_epi8(_mm256_xor_si256(a, bias), _mm256_xor_si256(b, bias)) }),
        (|a, b| { let bias = _mm256_set1_epi8(i8::MIN); _mm256_or_si256(_mm256_cmpgt_epi8(_mm256_xor_si256(b, bias), _mm256_xor_si256(a, bias)), _mm256_cmpeq_epi8(a, b)) }),
        (|a, b| { let bias = _mm256_set1_epi8(i8::MIN); _mm256_or_si256(_mm256_cmpgt_epi8(_mm256_xor_si256(a, bias), _mm256_xor_si256(b, bias)), _mm256_cmpeq_epi8(a, b)) }),
        (|a, b| _mm256_xor_si256(_mm256_cmpeq_epi8(a, b), _mm256_set1_epi8(-1)))
    );
    #[rustfmt::skip]
    impl_cmp!(U16x8, Mask16x8,
        _mm_cmpeq_epi16,
        (|a, b| { let bias = _mm_set1_epi16(i16::MIN); _mm_cmplt_epi16(_mm_xor_si128(a, bias), _mm_xor_si128(b, bias)) }),
        (|a, b| { let bias = _mm_set1_epi16(i16::MIN); _mm_cmpgt_epi16(_mm_xor_si128(a, bias), _mm_xor_si128(b, bias)) }),
        (|a, b| { let bias = _mm_set1_epi16(i16::MIN); _mm_or_si128(_mm_cmplt_epi16(_mm_xor_si128(a, bias), _mm_xor_si128(b, bias)), _mm_cmpeq_epi16(a, b)) }),
        (|a, b| { let bias = _mm_set1_epi16(i16::MIN); _mm_or_si128(_mm_cmpgt_epi16(_mm_xor_si128(a, bias), _mm_xor_si128(b, bias)), _mm_cmpeq_epi16(a, b)) }),
        (|a, b| _mm_xor_si128(_mm_cmpeq_epi16(a, b), _mm_set1_epi16(-1)))
    );
    #[rustfmt::skip]
    impl_cmp!(U16x16, Mask16x16,
        _mm256_cmpeq_epi16,
        (|a, b| { let bias = _mm256_set1_epi16(i16::MIN); _mm256_cmpgt_epi16(_mm256_xor_si256(b, bias), _mm256_xor_si256(a, bias)) }),
        (|a, b| { let bias = _mm256_set1_epi16(i16::MIN); _mm256_cmpgt_epi16(_mm256_xor_si256(a, bias), _mm256_xor_si256(b, bias)) }),
        (|a, b| { let bias = _mm256_set1_epi16(i16::MIN); _mm256_or_si256(_mm256_cmpgt_epi16(_mm256_xor_si256(b, bias), _mm256_xor_si256(a, bias)), _mm256_cmpeq_epi16(a, b)) }),
        (|a, b| { let bias = _mm256_set1_epi16(i16::MIN); _mm256_or_si256(_mm256_cmpgt_epi16(_mm256_xor_si256(a, bias), _mm256_xor_si256(b, bias)), _mm256_cmpeq_epi16(a, b)) }),
        (|a, b| _mm256_xor_si256(_mm256_cmpeq_epi16(a, b), _mm256_set1_epi16(-1)))
    );
    #[rustfmt::skip]
    impl_cmp!(U32x4, Mask32x4,
        _mm_cmpeq_epi32,
        (|a, b| { let bias = _mm_set1_epi32(i32::MIN); _mm_cmplt_epi32(_mm_xor_si128(a, bias), _mm_xor_si128(b, bias)) }),
        (|a, b| { let bias = _mm_set1_epi32(i32::MIN); _mm_cmpgt_epi32(_mm_xor_si128(a, bias), _mm_xor_si128(b, bias)) }),
        (|a, b| { let bias = _mm_set1_epi32(i32::MIN); _mm_or_si128(_mm_cmplt_epi32(_mm_xor_si128(a, bias), _mm_xor_si128(b, bias)), _mm_cmpeq_epi32(a, b)) }),
        (|a, b| { let bias = _mm_set1_epi32(i32::MIN); _mm_or_si128(_mm_cmpgt_epi32(_mm_xor_si128(a, bias), _mm_xor_si128(b, bias)), _mm_cmpeq_epi32(a, b)) }),
        (|a, b| _mm_xor_si128(_mm_cmpeq_epi32(a, b), _mm_set1_epi32(-1)))
    );
    #[rustfmt::skip]
    impl_cmp!(U32x8, Mask32x8,
        _mm256_cmpeq_epi32,
        (|a, b| { let bias = _mm256_set1_epi32(i32::MIN); _mm256_cmpgt_epi32(_mm256_xor_si256(b, bias), _mm256_xor_si256(a, bias)) }),
        (|a, b| { let bias = _mm256_set1_epi32(i32::MIN); _mm256_cmpgt_epi32(_mm256_xor_si256(a, bias), _mm256_xor_si256(b, bias)) }),
        (|a, b| { let bias = _mm256_set1_epi32(i32::MIN); _mm256_or_si256(_mm256_cmpgt_epi32(_mm256_xor_si256(b, bias), _mm256_xor_si256(a, bias)), _mm256_cmpeq_epi32(a, b)) }),
        (|a, b| { let bias = _mm256_set1_epi32(i32::MIN); _mm256_or_si256(_mm256_cmpgt_epi32(_mm256_xor_si256(a, bias), _mm256_xor_si256(b, bias)), _mm256_cmpeq_epi32(a, b)) }),
        (|a, b| _mm256_xor_si256(_mm256_cmpeq_epi32(a, b), _mm256_set1_epi32(-1)))
    );
    #[rustfmt::skip]
    impl_cmp!(U64x2, Mask64x2,
        _mm_cmpeq_epi64,
        (|a, b| { let bias = _mm_set1_epi64x(i64::MIN); _mm_cmpgt_epi64(_mm_xor_si128(b, bias), _mm_xor_si128(a, bias)) }),
        (|a, b| { let bias = _mm_set1_epi64x(i64::MIN); _mm_cmpgt_epi64(_mm_xor_si128(a, bias), _mm_xor_si128(b, bias)) }),
        (|a, b| { let bias = _mm_set1_epi64x(i64::MIN); _mm_or_si128(_mm_cmpgt_epi64(_mm_xor_si128(b, bias), _mm_xor_si128(a, bias)), _mm_cmpeq_epi64(a, b)) }),
        (|a, b| { let bias = _mm_set1_epi64x(i64::MIN); _mm_or_si128(_mm_cmpgt_epi64(_mm_xor_si128(a, bias), _mm_xor_si128(b, bias)), _mm_cmpeq_epi64(a, b)) }),
        (|a, b| _mm_xor_si128(_mm_cmpeq_epi64(a, b), _mm_set1_epi64x(-1)))
    );
    #[rustfmt::skip]
    impl_cmp!(U64x4, Mask64x4,
        _mm256_cmpeq_epi64,
        (|a, b| { let bias = _mm256_set1_epi64x(i64::MIN); _mm256_cmpgt_epi64(_mm256_xor_si256(b, bias), _mm256_xor_si256(a, bias)) }),
        (|a, b| { let bias = _mm256_set1_epi64x(i64::MIN); _mm256_cmpgt_epi64(_mm256_xor_si256(a, bias), _mm256_xor_si256(b, bias)) }),
        (|a, b| { let bias = _mm256_set1_epi64x(i64::MIN); _mm256_or_si256(_mm256_cmpgt_epi64(_mm256_xor_si256(b, bias), _mm256_xor_si256(a, bias)), _mm256_cmpeq_epi64(a, b)) }),
        (|a, b| { let bias = _mm256_set1_epi64x(i64::MIN); _mm256_or_si256(_mm256_cmpgt_epi64(_mm256_xor_si256(a, bias), _mm256_xor_si256(b, bias)), _mm256_cmpeq_epi64(a, b)) }),
        (|a, b| _mm256_xor_si256(_mm256_cmpeq_epi64(a, b), _mm256_set1_epi64x(-1)))
    );
}

#[cfg(target_arch = "aarch64")]
pub mod arch {}
