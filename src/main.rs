mod error;

use crate::error::{AppError, Result};
use ash::khr::{surface, swapchain};
use ash::{Entry, ext::debug_utils, vk};
use std::{default::Default, sync::Arc};
use winit::dpi::PhysicalSize;
#[allow(deprecated)]
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    raw_window_handle::{HasRawDisplayHandle, HasRawWindowHandle},
    window::{Window, WindowId},
};

pub struct VulkanContext {
    pub entry: ash::Entry,
    pub instance: ash::Instance,
    pub surface_loader: ash::khr::surface::Instance,
    pub surface: vk::SurfaceKHR,
    pub physical_device: vk::PhysicalDevice,
    pub device: ash::Device,
    pub graphics_queue: vk::Queue,
    pub graphics_queue_family_index: u32,
}

impl VulkanContext {
    pub fn new(window: &Window) -> Result<Self> {
        let entry = Entry::linked();

        #[allow(deprecated)]
        let display_handle = window.raw_display_handle().unwrap();
        #[allow(deprecated)]
        let window_handle = window.raw_window_handle().unwrap();

        let mut extension_names =
            ash_window::enumerate_required_extensions(display_handle)?.to_vec();
        extension_names.push(debug_utils::NAME.as_ptr());

        let app_info = vk::ApplicationInfo::default()
            .application_name(c"Renderer")
            .api_version(vk::API_VERSION_1_3);

        let instance_create_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_extension_names(extension_names.as_slice());

        let instance = unsafe { entry.create_instance(&instance_create_info, None) }?;

        let surface = unsafe {
            ash_window::create_surface(&entry, &instance, display_handle, window_handle, None)
        }?;

        let surface_loader = surface::Instance::new(&entry, &instance);

        let physical_devices = unsafe { instance.enumerate_physical_devices() }?;
        let (physical_device, queue_family_index) = physical_devices
            .iter()
            .filter_map(|&physical_device| {
                let properties =
                    unsafe { instance.get_physical_device_properties(physical_device) };

                let queue_family_index = unsafe {
                    instance.get_physical_device_queue_family_properties(physical_device)
                }
                .iter()
                .enumerate()
                .find_map(|(index, info)| {
                    let index = index as u32;
                    let has_graphics = info.queue_flags.contains(vk::QueueFlags::GRAPHICS);
                    let has_surface = unsafe {
                        surface_loader.get_physical_device_surface_support(
                            physical_device,
                            index,
                            surface,
                        )
                    }
                    .unwrap_or(false);

                    if has_graphics && has_surface {
                        Some(index)
                    } else {
                        None
                    }
                })?;

                // TODO: score better
                let mut score = 0;
                score += match properties.device_type {
                    vk::PhysicalDeviceType::DISCRETE_GPU => 1_000_000,
                    vk::PhysicalDeviceType::INTEGRATED_GPU => 100_000,
                    vk::PhysicalDeviceType::VIRTUAL_GPU => 10_000,
                    vk::PhysicalDeviceType::CPU => 1_000,
                    _ => 0,
                };

                println!(
                    "device: {}, score: {}",
                    properties.device_name_as_c_str().unwrap().to_str().unwrap(),
                    score
                );

                Some((score, (physical_device, queue_family_index)))
            })
            .max_by_key(|(score, _)| *score)
            .map(|(_, device_info)| device_info)
            .expect("Couldn't find suitable device");

        let priorities: [f32; _] = [1.0];

        let queue_info = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family_index)
            .queue_priorities(&priorities);

        let device_extension_names_raw = [swapchain::NAME.as_ptr()];

        let features = vk::PhysicalDeviceFeatures {
            shader_clip_distance: 1,
            ..Default::default()
        };

        let device_create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(std::slice::from_ref(&queue_info))
            .enabled_extension_names(&device_extension_names_raw)
            .enabled_features(&features);

        let device = unsafe { instance.create_device(physical_device, &device_create_info, None)? };

        let graphics_queue = unsafe { device.get_device_queue(queue_family_index, 0) };

        Ok(Self {
            entry,
            instance,
            surface_loader,
            surface,
            physical_device,
            device,
            graphics_queue,
            graphics_queue_family_index: queue_family_index,
        })
    }

    pub fn destroy(self) {
        unsafe {
            self.device.device_wait_idle().unwrap();

            self.device.destroy_device(None);
            self.surface_loader.destroy_surface(self.surface, None);
            self.instance.destroy_instance(None);
        }
    }
}

pub struct SwapchainContext {
    pub swapchain_loader: ash::khr::swapchain::Device,
    pub images: Vec<vk::Image>,
    pub image_views: Vec<vk::ImageView>,
    pub extent: vk::Extent2D,
    pub format: vk::Format,
    swapchain: vk::SwapchainKHR,
}

impl SwapchainContext {
    pub fn new(context: &VulkanContext, window_size: PhysicalSize<u32>) -> Result<Self> {
        let swapchain_loader = swapchain::Device::new(&context.instance, &context.device);

        let surface_format = unsafe {
            context
                .surface_loader
                .get_physical_device_surface_formats(context.physical_device, context.surface)?[0]
        };

        let surface_capabilities = unsafe {
            context
                .surface_loader
                .get_physical_device_surface_capabilities(
                    context.physical_device,
                    context.surface,
                )?
        };

        let mut desired_image_count = surface_capabilities.min_image_count + 1;
        if surface_capabilities.max_image_count > 0
            && desired_image_count > surface_capabilities.max_image_count
        {
            desired_image_count = surface_capabilities.max_image_count;
        }

        let extent = match surface_capabilities.current_extent.width {
            u32::MAX => vk::Extent2D {
                width: window_size.width,
                height: window_size.height,
            },
            _ => surface_capabilities.current_extent,
        };

        let pre_transform = if surface_capabilities
            .supported_transforms
            .contains(vk::SurfaceTransformFlagsKHR::IDENTITY)
        {
            vk::SurfaceTransformFlagsKHR::IDENTITY
        } else {
            surface_capabilities.current_transform
        };

        let present_modes = unsafe {
            context
                .surface_loader
                .get_physical_device_surface_present_modes(context.physical_device, context.surface)
        }?;

        let present_mode = present_modes
            .iter()
            .cloned()
            .find(|&mode| mode == vk::PresentModeKHR::MAILBOX)
            .unwrap_or(vk::PresentModeKHR::FIFO);

        let swapchain_create_info = vk::SwapchainCreateInfoKHR::default()
            .surface(context.surface)
            .min_image_count(desired_image_count)
            .image_color_space(surface_format.color_space)
            .image_format(surface_format.format)
            .image_extent(extent)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(pre_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(present_mode)
            .clipped(true)
            .image_array_layers(1);

        let swapchain = unsafe { swapchain_loader.create_swapchain(&swapchain_create_info, None)? };

        let images = unsafe { swapchain_loader.get_swapchain_images(swapchain) }?;
        let image_views: Vec<vk::ImageView> = images
            .iter()
            .map(|&image| {
                let create_view_info = vk::ImageViewCreateInfo::default()
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(surface_format.format)
                    .components(vk::ComponentMapping {
                        r: vk::ComponentSwizzle::R,
                        g: vk::ComponentSwizzle::G,
                        b: vk::ComponentSwizzle::B,
                        a: vk::ComponentSwizzle::A,
                    })
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    })
                    .image(image);
                unsafe {
                    context
                        .device
                        .create_image_view(&create_view_info, None)
                        .unwrap()
                }
            })
            .collect();

        Ok(Self {
            swapchain_loader,
            swapchain,
            images,
            image_views,
            extent,
            format: surface_format.format,
        })
    }

    pub fn destroy(self, device: &ash::Device) {
        unsafe {
            device.device_wait_idle().unwrap();

            self.image_views.iter().for_each(|&image_view| {
                device.destroy_image_view(image_view, None);
            });

            self.swapchain_loader
                .destroy_swapchain(self.swapchain, None);
        }
    }
}

pub struct FrameData {
    pub command_pool: vk::CommandPool,
    pub main_command_buffer: vk::CommandBuffer,
    pub image_available_semaphore: vk::Semaphore,
    pub render_finished_semaphore: vk::Semaphore,
    pub render_fence: vk::Fence,
}

impl FrameData {
    pub fn new(context: &VulkanContext) -> Result<Self> {
        let device = &context.device;

        let pool_create_info = vk::CommandPoolCreateInfo::default()
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            .queue_family_index(context.graphics_queue_family_index);

        let pool = unsafe { device.create_command_pool(&pool_create_info, None)? };

        let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::default()
            .command_buffer_count(1)
            .command_pool(pool)
            .level(vk::CommandBufferLevel::PRIMARY);

        let command_buffer =
            unsafe { device.allocate_command_buffers(&command_buffer_allocate_info)? }[0];

        let semaphore_info = vk::SemaphoreCreateInfo::default();

        let image_available_semaphore = unsafe { device.create_semaphore(&semaphore_info, None)? };
        let render_finished_semaphore = unsafe { device.create_semaphore(&semaphore_info, None)? };

        let fence_info = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);

        let render_fence = unsafe { device.create_fence(&fence_info, None)? };

        Ok(Self {
            command_pool: pool,
            main_command_buffer: command_buffer,
            image_available_semaphore,
            render_finished_semaphore,
            render_fence,
        })
    }

    pub fn destroy(self, device: &ash::Device) {
        unsafe {
            device.device_wait_idle().unwrap();

            device.destroy_fence(self.render_fence, None);
            device.destroy_semaphore(self.render_finished_semaphore, None);
            device.destroy_semaphore(self.image_available_semaphore, None);
            device.destroy_command_pool(self.command_pool, None);
        }
    }
}

pub struct State {
    window: Arc<Window>,
    vulkan_context: VulkanContext,
    swapchain_context: SwapchainContext,
    frames: [FrameData; 2],
}

impl State {
    pub fn new(window: Arc<Window>) -> Result<Self> {
        let window_size = window.inner_size();

        let vulkan_context = VulkanContext::new(&window)?;
        let swapchain_context = SwapchainContext::new(&vulkan_context, window_size)?;

        let frames = [
            FrameData::new(&vulkan_context)?,
            FrameData::new(&vulkan_context)?,
        ];

        Ok(Self {
            window,
            vulkan_context,
            swapchain_context,
            frames,
        })
    }

    pub fn resize(&mut self, _width: u32, _height: u32) {}

    pub fn handle_key(&mut self, event_loop: &ActiveEventLoop, code: KeyCode, is_pressed: bool) {
        match (code, is_pressed) {
            (KeyCode::Escape, true) => event_loop.exit(),
            _ => (),
        }
    }

    pub fn render(&mut self) -> Result<()> {
        self.window.request_redraw();

        Ok(())
    }

    pub fn update(&mut self) {
        //
    }
}

#[derive(Default)]
struct App {
    state: Option<State>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window_attrs = Window::default_attributes().with_title("Map Renderer");
        let window = Arc::new(event_loop.create_window(window_attrs).unwrap());

        self.state = Some(State::new(window).unwrap());
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let Some(state) = &mut self.state else { return };

        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(size) => state.resize(size.width, size.height),
            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(code) = event.physical_key {
                    state.handle_key(event_loop, code, event.state.is_pressed());
                }
            }
            WindowEvent::RedrawRequested => {
                state.update();
                if let Err(e) = state.render() {
                    log::error!("{e}");
                    event_loop.exit();
                }
            }
            _ => (),
        }
    }

    fn exiting(&mut self, _event_loop: &ActiveEventLoop) {
        let Some(state) = self.state.take() else {
            return;
        };

        let device = &state.vulkan_context.device;

        unsafe { device.device_wait_idle().unwrap() };
        state
            .frames
            .into_iter()
            .for_each(|frame| frame.destroy(&device));

        state.swapchain_context.destroy(&device);
        state.vulkan_context.destroy();
        println!("Hello");
    }
}

fn main() {
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Wait);
    let mut app = App::default();
    event_loop.run_app(&mut app).unwrap();
}
