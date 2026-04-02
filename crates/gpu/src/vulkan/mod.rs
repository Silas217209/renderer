use crate::core::{Adapter, Backend, Device, Instance};
use crate::util::InstanceDescriptor;
use ash::vk;
use raw_window_handle::{HasDisplayHandle, HasRawDisplayHandle};
use std::borrow::Cow;
use std::ffi;

#[derive(Clone)]
pub struct VulkanBackend;

impl Backend for VulkanBackend {
    type Instance = VulkanInstance;
    type Adapter = VulkanAdapter;
    type Device = VulkanDevice;
}

pub struct VulkanInstance {
    instance: ash::Instance,
}
impl Instance<VulkanBackend> for VulkanInstance {
    fn new(descriptor: &InstanceDescriptor) -> Self {
        let entry = unsafe { ash::Entry::load() }.expect("Failed to find Vulkan loader");

        let app_info = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_3);

        let required_extensions = if let Some(handle) = descriptor.display_handle {
            ash_window::enumerate_required_extensions(handle.raw_display_handle().unwrap()).unwrap()
        } else {
            &[]
        };

        let create_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_extension_names(required_extensions);

        let instance =
            unsafe { entry.create_instance(&create_info, None) }.expect("Instance cration error");

        Self { instance }
    }
}

pub struct VulkanAdapter;
impl Adapter<VulkanBackend> for VulkanAdapter {}

pub struct VulkanDevice;
impl Device<VulkanBackend> for VulkanDevice {}

unsafe extern "system" fn vulkan_debug_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    _user_data: *mut std::os::raw::c_void,
) -> vk::Bool32 {
    let callback_data = *p_callback_data;
    let message_id_number = callback_data.message_id_number;

    let message_id_name = if callback_data.p_message_id_name.is_null() {
        Cow::from("")
    } else {
        unsafe { ffi::CStr::from_ptr(callback_data.p_message_id_name) }.to_string_lossy()
    };

    let message = if callback_data.p_message.is_null() {
        Cow::from("")
    } else {
        unsafe { ffi::CStr::from_ptr(callback_data.p_message) }.to_string_lossy()
    };

    println!(
        "{message_severity:?}:\n{message_type:?} [{message_id_name} ({message_id_number})] : {message}\n"
    );

    vk::FALSE
}
