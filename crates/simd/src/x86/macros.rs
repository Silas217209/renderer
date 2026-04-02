#[macro_export]
macro_rules! impl_intr {
    ($fn_name:ident(self $(, $arg:ident)*) => $intrinsic:expr) => {
        pub fn $fn_name(self $(, $arg: Self)*) -> Self {
            unsafe { Self($intrinsic(self.0 $(, $arg.0)*)) }
        }
    };
}

#[macro_export]
macro_rules! impl_init {
    (
        $type:ident,
        $lanes:literal,
        $zero:ident,
        $splat:ident,
        $load:ident,
        $load_aligned:ident,
        $store:ident,
        $store_aligned:ident
    ) => {
        pub fn new(data: [$type; $lanes]) -> Self {
            unsafe { Self($load(data.as_ptr())) }
        }

        pub fn zero() -> Self {
            unsafe { Self($zero()) }
        }

        pub fn splat(val: $type) -> Self {
            unsafe { Self($splat(val)) }
        }

        pub unsafe fn load_ptr(ptr: *const $type) -> Self {
            unsafe { Self($load(ptr)) }
        }
        pub unsafe fn load_aligned_ptr(ptr: *const $type) -> Self {
            unsafe { Self($load_aligned(ptr)) }
        }

        pub unsafe fn store_ptr(self, ptr: *mut $type) {
            unsafe { $store(ptr, self.0) }
        }

        pub unsafe fn store_aligned_ptr(self, ptr: *mut $type) {
            unsafe { $store_aligned(ptr, self.0) }
        }

        pub fn load_slice(data: &[$type]) -> Self {
            assert!(data.len() >= $lanes);
            unsafe { Self($load(data.as_ptr())) }
        }

        pub fn store_slice(self, slice: &mut [$type]) {
            unsafe { $store(slice.as_mut_ptr(), self.0) }
        }
    };
}
