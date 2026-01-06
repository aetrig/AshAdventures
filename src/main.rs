#![allow(unused)]
#![allow(dead_code)]
use ash::{
    Entry,
    ext::{self, image_compression_control},
    khr::{get_surface_capabilities2, surface},
    vk::{self, Handle, PhysicalDevice},
};
use glfw::{self, PWindow};
use glm::clamp;
use std::{
    cmp::max,
    ffi::{CStr, CString},
    ops::Deref,
    ptr::{self, null},
};

fn main() {
    let mut app = VulkanRenderer::new();

    app.run();
}

const WIDTH: u32 = 800;
const HEIGHT: u32 = 600;

const VALIDATION_LAYERS: [&str; 1] = ["VK_LAYER_KHRONOS_validation"];
#[cfg(debug_assertions)]
const VALIDATION_LAYERS_ENABLED: bool = true;
#[cfg(not(debug_assertions))]
const VALIDATION_LAYERS_ENABLED: bool = false;

unsafe extern "system" fn debug_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _p_user_data: *mut std::ffi::c_void,
) -> vk::Bool32 {
    let message = unsafe { CStr::from_ptr((*p_callback_data).p_message) };
    let severity = format!("{:?}", message_severity);
    let ty = format!("{:?}", message_type);
    let severity_color = match message_severity {
        vk::DebugUtilsMessageSeverityFlagsEXT::INFO => "\x1b[36m",
        vk::DebugUtilsMessageSeverityFlagsEXT::ERROR => "\x1b[31m",
        vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE => "\x1b[92m",
        vk::DebugUtilsMessageSeverityFlagsEXT::WARNING => "\x1b[33m",
        _ => "\x1b[0m",
    };
    let type_color = match message_type {
        vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION => "\x1b[35m",
        vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE => "\x1b[93m",
        vk::DebugUtilsMessageTypeFlagsEXT::DEVICE_ADDRESS_BINDING => "\x1b[33m",
        vk::DebugUtilsMessageTypeFlagsEXT::GENERAL => "\x1b[32m",
        _ => "\x1b[0m",
    };
    println!(
        "\x1b[34m[DEBUG] {severity_color}[{}] {type_color}[{}]\x1b[0m   {}",
        severity,
        ty,
        message.to_str().expect("Failed to parse CStr")
    );

    vk::FALSE
}

struct VulkanRenderer {
    glfw: glfw::Glfw,
    window: glfw::PWindow,

    entry: ash::Entry,
    instance: ash::Instance,

    debug_instance: Option<ext::debug_utils::Instance>,
    debug_messenger: Option<vk::DebugUtilsMessengerEXT>,

    surface: vk::SurfaceKHR,
    surface_instance: ash::khr::surface::Instance,

    physical_device: vk::PhysicalDevice,
    device: ash::Device,

    graphics_family: u32,
    graphics_queue: vk::Queue,
    presentation_family: u32,
    presentation_queue: vk::Queue,

    swapchain_device: ash::khr::swapchain::Device,
    swapchain: vk::SwapchainKHR,
    swapchain_images: Vec<vk::Image>,
    swapchain_image_format: vk::Format,
    swapchain_extent: vk::Extent2D,
    swapchain_image_views: Vec<vk::ImageView>,
}

impl VulkanRenderer {
    // Initialization code
    pub fn new() -> Self {
        let (glfw, window) = VulkanRenderer::init_glfw_window();
        let (entry, instance) = VulkanRenderer::create_vk_instance(&glfw);
        let (debug_instance, debug_messenger) =
            VulkanRenderer::setup_debug_messenger(&entry, &instance);
        let (surface, surface_instance) =
            VulkanRenderer::create_surface(&window, &instance, &entry);
        let physical_device = VulkanRenderer::pick_physical_device(&instance);
        let (device, graphics_family, graphics_queue, presentation_family, presentation_queue) =
            VulkanRenderer::create_logical_device(
                &instance,
                &physical_device,
                &surface_instance,
                &surface,
            );
        let (
            swapchain_device,
            swapchain,
            swapchain_images,
            swapchain_image_format,
            swapchain_extent,
        ) = VulkanRenderer::create_swapchain(
            &surface_instance,
            &surface,
            &physical_device,
            &instance,
            &device,
            &window,
            graphics_family,
            presentation_family,
        );
        let swapchain_image_views =
            VulkanRenderer::create_image_views(swapchain_image_format, &swapchain_images, &device);
        VulkanRenderer {
            glfw,
            window,
            entry,
            instance,
            debug_instance,
            debug_messenger,
            surface,
            surface_instance,
            physical_device,
            device,
            graphics_family,
            graphics_queue,
            presentation_family,
            presentation_queue,
            swapchain_device,
            swapchain,
            swapchain_images,
            swapchain_image_format,
            swapchain_extent,
            swapchain_image_views,
        }
    }

    pub fn run(&mut self) {
        self.main_loop();
    }

    fn init_glfw_window() -> (glfw::Glfw, glfw::PWindow) {
        let mut glfw = glfw::init(glfw::fail_on_errors).unwrap();
        glfw.window_hint(glfw::WindowHint::ClientApi(glfw::ClientApiHint::NoApi));
        glfw.window_hint(glfw::WindowHint::Resizable(false));
        let (window, _) = glfw
            .create_window(WIDTH, HEIGHT, "Vulkan", glfw::WindowMode::Windowed)
            .expect("Failed to create GLFW window");
        (glfw, window)
    }

    fn create_vk_instance(glfw: &glfw::Glfw) -> (ash::Entry, ash::Instance) {
        let entry = Entry::linked();

        // Application info struct
        let app_name = CString::new("Hello Triangle").expect("Failed to create a CString");
        let engine_name = CString::new("No Engine").expect("Failed to create a CString");
        let app_info = vk::ApplicationInfo::default()
            .application_name(app_name.as_c_str())
            .api_version(vk::make_api_version(0, 1, 0, 0))
            .engine_name(engine_name.as_c_str())
            .engine_version(vk::make_api_version(0, 1, 0, 0))
            .api_version(vk::API_VERSION_1_3);

        // Validation Layers
        let mut required_layers: Vec<&str> = Vec::new();
        if VALIDATION_LAYERS_ENABLED {
            for layer in VALIDATION_LAYERS {
                required_layers.push(layer);
            }
        }
        let layer_properties = unsafe {
            entry
                .enumerate_instance_layer_properties()
                .expect("Failed to get available layers")
        }
        .iter()
        .map(|lay| {
            lay.layer_name_as_c_str()
                .expect("Failed to extract layer name")
                .to_str()
                .expect("Failed to convert to str")
                .to_string()
        })
        .collect::<Vec<_>>();

        // Checking if all required layers available
        for layer in &required_layers {
            if !layer_properties.contains(&layer.to_string()) {
                panic!("Required layer not supported! {layer}")
            }
        }

        // Converting layers to format applicable to Create info struct
        let layers_count = required_layers.len() as u32;

        let layers = required_layers
            .iter()
            .map(|name| CString::new(*name).expect("Failed to change extension names to CStrings"))
            .collect::<Vec<_>>();

        let layers = layers.iter().map(|name| name.as_ptr()).collect::<Vec<_>>();

        // Extensions required by GLFW
        let mut glfw_extensions = glfw
            .get_required_instance_extensions()
            .expect("Failed to get required GLFW extensions");

        let mut extensions: Vec<String> = Vec::new();
        extensions.append(&mut glfw_extensions);
        if VALIDATION_LAYERS_ENABLED {
            extensions.push(
                vk::EXT_DEBUG_UTILS_NAME
                    .to_str()
                    .expect("Failed to convert to str")
                    .to_string(),
            );
        }

        // Available Extensions
        let extension_properties = unsafe {
            entry
                .enumerate_instance_extension_properties(None)
                .expect("Failed to get available extensions")
        }
        .iter()
        .map(|ex| {
            ex.extension_name_as_c_str()
                .expect("Failed to extract extension name")
                .to_str()
                .expect("Failed to convert to str")
                .to_string()
        })
        .collect::<Vec<_>>();

        // Checking if all required extensions available
        for extension in &extensions {
            if !extension_properties.contains(&extension) {
                panic!("Required extension not supported! {extension}")
            }
        }

        // Converting extensions to format applicable to Create info struct
        let extensions_count = extensions.len() as u32;

        let extensions = extensions
            .iter()
            .map(|name| {
                CString::new(name.as_str()).expect("Failed to change extension names to CStrings")
            })
            .collect::<Vec<_>>();

        let extensions = extensions
            .iter()
            .map(|name| name.as_ptr())
            .collect::<Vec<_>>();

        // Create info struct
        let mut create_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_layer_names(layers.as_slice())
            .enabled_extension_names(extensions.as_slice());

        // Creating instance
        (entry.clone(), unsafe {
            entry
                .create_instance(&create_info, None)
                .expect("Failed to create a Vulkan Instance")
        })
    }

    fn setup_debug_messenger(
        entry: &ash::Entry,
        instance: &ash::Instance,
    ) -> (
        Option<ext::debug_utils::Instance>,
        Option<vk::DebugUtilsMessengerEXT>,
    ) {
        if !VALIDATION_LAYERS_ENABLED {
            return (None, None);
        } else {
            let severity_flags = vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE
                | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                | vk::DebugUtilsMessageSeverityFlagsEXT::INFO
                | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING;
            let message_type_flags = vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE
                | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION;

            let debug_utils_create_info = vk::DebugUtilsMessengerCreateInfoEXT::default()
                .message_severity(severity_flags)
                .message_type(message_type_flags)
                .pfn_user_callback(Some(debug_callback));

            let debug_instance = ext::debug_utils::Instance::new(entry, instance);

            let debug_messenger = unsafe {
                debug_instance.create_debug_utils_messenger(&debug_utils_create_info, None)
            }
            .expect("Failed to create debug messenger");

            (Some(debug_instance), Some(debug_messenger))
        }
    }

    fn pick_physical_device(instance: &ash::Instance) -> vk::PhysicalDevice {
        let devices = unsafe {
            instance
                .enumerate_physical_devices()
                .expect("Failed to get physical devices")
        };

        // We want the device to support these extensions
        let device_extensions = vec![
            vk::KHR_SWAPCHAIN_NAME,
            vk::KHR_SPIRV_1_4_NAME,
            vk::KHR_SYNCHRONIZATION2_NAME,
            vk::KHR_CREATE_RENDERPASS2_NAME,
        ];

        let mut physical_device: Option<vk::PhysicalDevice> = None;
        // Finding suitable devices
        devices.iter().find(|device| {
            let queue_families =
                unsafe { instance.get_physical_device_queue_family_properties(**device) };

            // We want Vulkan version support at least 1.3
            let mut is_suitable = unsafe {
                instance
                    .get_physical_device_properties(**device)
                    .api_version
            } >= vk::API_VERSION_1_3;

            // We want access to a graphics queue
            let qfp_iter = queue_families
                .iter()
                .find(|qfp| qfp.queue_flags & vk::QueueFlags::GRAPHICS != vk::QueueFlags::empty());
            is_suitable = is_suitable && qfp_iter.is_some();

            let extensions = unsafe { instance.enumerate_device_extension_properties(**device) }
                .expect("Failed to get device extensions");
            let mut found = true;
            for extension in &device_extensions {
                let ex_iter = extensions.iter().find(|ex| {
                    ex.extension_name_as_c_str()
                        .expect("Failed to get extension name")
                        == extension
                });
                found = found && ex_iter.is_some();
            }
            is_suitable = is_suitable && found;
            if is_suitable {
                physical_device = Some(**device);
            }
            is_suitable
        });

        physical_device.expect("No suitable devices found")
    }

    fn create_logical_device(
        instance: &ash::Instance,
        physical_device: &vk::PhysicalDevice,
        surface_instance: &ash::khr::surface::Instance,
        surface: &vk::SurfaceKHR,
    ) -> (ash::Device, u32, vk::Queue, u32, vk::Queue) {
        let queue_family_properties =
            unsafe { instance.get_physical_device_queue_family_properties(*physical_device) };

        // Find graphics queue
        let mut graphics_index = queue_family_properties
            .iter()
            .position(|qfp| qfp.queue_flags & vk::QueueFlags::GRAPHICS != vk::QueueFlags::empty())
            .expect("Failed to find a graphics queue") as u32;

        // Check if it's suitable for presentation as well
        let mut present_index = if unsafe {
            surface_instance.get_physical_device_surface_support(
                *physical_device,
                graphics_index,
                *surface,
            )
        }
        .expect("Failed to check for presentation support")
        {
            graphics_index
        } else {
            queue_family_properties.len() as u32
        };

        // graphics_index doesn't support present -> look for another index that supports both
        if present_index == queue_family_properties.len() as u32 {
            let potential_index =
                queue_family_properties
                    .iter()
                    .enumerate()
                    .position(|(idx, qfp)| {
                        qfp.queue_flags & vk::QueueFlags::GRAPHICS != vk::QueueFlags::empty()
                            && unsafe {
                                surface_instance
                                    .get_physical_device_surface_support(
                                        *physical_device,
                                        idx as u32,
                                        *surface,
                                    )
                                    .expect("Failed to check for presentation support")
                            }
                    });

            // There is no queue that supports both graphics and present -> Get them separately
            match potential_index {
                Some(idx) => {
                    graphics_index = idx as u32;
                    present_index = idx as u32;
                }
                None => {
                    present_index = queue_family_properties
                        .iter()
                        .enumerate()
                        .position(|(idx, _)| {
                            unsafe {
                                surface_instance.get_physical_device_surface_support(
                                    *physical_device,
                                    idx as u32,
                                    *surface,
                                )
                            }
                            .expect("Failed to check for presentation support")
                        })
                        .expect("Couldn't find a presentation queue")
                        as u32
                }
            }
        }

        let queue_priorities = [0.5f32];
        let device_queue_create_info = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(graphics_index)
            .queue_priorities(&queue_priorities);

        let mut device_features = vk::PhysicalDeviceFeatures2::default();
        let mut vulkan13_features =
            vk::PhysicalDeviceVulkan13Features::default().dynamic_rendering(true);
        let mut extended_features = vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT::default()
            .extended_dynamic_state(true);
        device_features.push_next(&mut vulkan13_features);
        device_features.push_next(&mut extended_features);

        let device_extensions = vec![
            vk::KHR_SWAPCHAIN_NAME,
            vk::KHR_SPIRV_1_4_NAME,
            vk::KHR_SYNCHRONIZATION2_NAME,
            vk::KHR_CREATE_RENDERPASS2_NAME,
        ]
        .iter()
        .map(|ex| ex.as_ptr())
        .collect::<Vec<_>>();

        let queue_create_infos = [device_queue_create_info];
        let mut device_create_info = vk::DeviceCreateInfo::default()
            .push_next(&mut device_features)
            .queue_create_infos(&queue_create_infos)
            .enabled_extension_names(device_extensions.as_slice());

        let device = unsafe { instance.create_device(*physical_device, &device_create_info, None) }
            .expect("Failed to create a logical device");

        (
            device.clone(),
            graphics_index,
            unsafe { device.get_device_queue(graphics_index, 0) },
            present_index,
            unsafe { device.get_device_queue(present_index, 0) },
        )
    }

    fn create_surface(
        window: &glfw::PWindow,
        instance: &ash::Instance,
        entry: &ash::Entry,
    ) -> (vk::SurfaceKHR, ash::khr::surface::Instance) {
        let mut glfw_surface: std::mem::MaybeUninit<vk::SurfaceKHR> =
            std::mem::MaybeUninit::uninit();
        if unsafe {
            window.create_window_surface(
                instance.handle().as_raw() as _,
                ptr::null(),
                glfw_surface.as_mut_ptr() as _,
            )
        } != vk::Result::SUCCESS.as_raw()
        {
            panic!("Failed to create a window surface");
        }

        (
            unsafe { glfw_surface.assume_init() },
            ash::khr::surface::Instance::new(entry, instance),
        )
    }

    fn create_swapchain(
        surface_instance: &ash::khr::surface::Instance,
        surface: &vk::SurfaceKHR,
        physical_device: &vk::PhysicalDevice,
        instance: &ash::Instance,
        device: &ash::Device,
        window: &PWindow,
        graphics_family: u32,
        presentation_family: u32,
    ) -> (
        ash::khr::swapchain::Device,
        vk::SwapchainKHR,
        Vec<vk::Image>,
        vk::Format,
        vk::Extent2D,
    ) {
        let surface_capabilities = unsafe {
            surface_instance.get_physical_device_surface_capabilities(*physical_device, *surface)
        }
        .expect("Failed to get surface capabilities");

        let swapchain_surface_format = VulkanRenderer::choose_swapchain_format(
            &unsafe {
                surface_instance.get_physical_device_surface_formats(*physical_device, *surface)
            }
            .expect("Failed to get available surface formats"),
        );

        let swapchain_extent = VulkanRenderer::choose_swap_extent(&surface_capabilities, window);

        let mut min_image_count = max(3u32, surface_capabilities.min_image_count);
        min_image_count = if surface_capabilities.max_image_count > 0
            && min_image_count > surface_capabilities.max_image_count
        {
            surface_capabilities.max_image_count
        } else {
            min_image_count
        };

        let present_mode = VulkanRenderer::choose_swap_present_mode(
            &unsafe {
                surface_instance
                    .get_physical_device_surface_present_modes(*physical_device, *surface)
            }
            .expect("Failed to get surface present modes"),
        );

        let mut swapchain_create_info = vk::SwapchainCreateInfoKHR::default()
            .flags(vk::SwapchainCreateFlagsKHR::default())
            .surface(*surface)
            .min_image_count(min_image_count)
            .image_format(swapchain_surface_format.format)
            .image_color_space(swapchain_surface_format.color_space)
            .image_extent(swapchain_extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(surface_capabilities.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(present_mode)
            .clipped(true)
            .old_swapchain(vk::SwapchainKHR::null());

        let queue_family_indices = [graphics_family, presentation_family];
        if graphics_family != presentation_family {
            swapchain_create_info = swapchain_create_info
                .image_sharing_mode(vk::SharingMode::CONCURRENT)
                .queue_family_indices(queue_family_indices.as_slice());
        };

        let swapchain_device = ash::khr::swapchain::Device::new(instance, device);
        let swapchain = unsafe { swapchain_device.create_swapchain(&swapchain_create_info, None) }
            .expect("Failed to create a swapchain");
        let swapchain_images = unsafe { swapchain_device.get_swapchain_images(swapchain) }
            .expect("Failed to get swapchain images");

        (
            swapchain_device,
            swapchain,
            swapchain_images,
            swapchain_surface_format.format,
            swapchain_extent,
        )
    }

    fn choose_swapchain_format(
        available_formats: &Vec<vk::SurfaceFormatKHR>,
    ) -> vk::SurfaceFormatKHR {
        for format in available_formats {
            if format.format == vk::Format::B8G8R8A8_SRGB
                && format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
            {
                return *format;
            }
        }

        *available_formats
            .get(0)
            .expect("No available surface formats")
    }

    fn choose_swap_present_mode(
        available_present_modes: &Vec<vk::PresentModeKHR>,
    ) -> vk::PresentModeKHR {
        for present_mode in available_present_modes {
            if *present_mode == vk::PresentModeKHR::MAILBOX {
                return *present_mode;
            }
        }
        vk::PresentModeKHR::FIFO
    }

    fn choose_swap_extent(
        capabilities: &vk::SurfaceCapabilitiesKHR,
        window: &glfw::PWindow,
    ) -> vk::Extent2D {
        if capabilities.current_extent.width != u32::MAX {
            return capabilities.current_extent;
        }
        let (mut width, mut height) = window.get_framebuffer_size();
        (width, height) = (
            clamp(
                width,
                capabilities.min_image_extent.width as i32,
                capabilities.max_image_extent.width as i32,
            ),
            clamp(
                height,
                capabilities.min_image_extent.height as i32,
                capabilities.max_image_extent.height as i32,
            ),
        );

        vk::Extent2D::default()
            .width(width as u32)
            .height(height as u32)
    }

    fn create_image_views(
        swapchain_image_format: vk::Format,
        swapchain_images: &Vec<vk::Image>,
        device: &ash::Device,
    ) -> Vec<vk::ImageView> {
        let mut image_view_create_info = vk::ImageViewCreateInfo::default()
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(swapchain_image_format)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_mip_level(0)
                    .level_count(1)
                    .base_array_layer(0)
                    .layer_count(1),
            );
        let mut swapchain_image_views: Vec<vk::ImageView> = Vec::new();
        for image in swapchain_images {
            swapchain_image_views.push(
                unsafe { device.create_image_view(&image_view_create_info.image(*image), None) }
                    .expect("Failed to create an image view"),
            );
        }
        swapchain_image_views
    }

    fn main_loop(&mut self) {
        while !self.window.should_close() {
            self.glfw.poll_events();
        }
    }
}

impl Drop for VulkanRenderer {
    // Cleanup code
    fn drop(&mut self) {
        unsafe {
            for image_view in &self.swapchain_image_views {
                self.device.destroy_image_view(*image_view, None);
            }
            self.swapchain_device
                .destroy_swapchain(self.swapchain, None);
            self.device.destroy_device(None);
            self.surface_instance.destroy_surface(self.surface, None);
            if VALIDATION_LAYERS_ENABLED {
                self.debug_instance
                    .as_ref()
                    .unwrap()
                    .destroy_debug_utils_messenger(self.debug_messenger.unwrap(), None);
            }
            self.instance.destroy_instance(None);
        };
    }
}
