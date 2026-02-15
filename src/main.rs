use ash::{
    Entry,
    ext::{self},
    vk::{self, Handle},
};
use glfw::{self, PWindow};
use glm::{self, clamp};
use std::{
    cmp::max,
    ffi::{CStr, CString},
    fs,
    mem::offset_of,
    os::raw::c_void,
    process::Command,
    ptr::{self},
    thread::current,
    time::{Duration, Instant},
};

fn main() {
    println!("Compiling shaders...");
    let _shader_compilation_output = Command::new("./shaders/compile.bat").arg("shader").output();
    println!("Shader compilation finished");

    let mut app = VulkanRenderer::new();

    app.run();
}

const WIDTH: u32 = 800;
const HEIGHT: u32 = 600;

const DEVICE_EXTENSIONS: [&CStr; 4] = [
    vk::KHR_SWAPCHAIN_NAME,
    vk::KHR_SPIRV_1_4_NAME,
    vk::KHR_SYNCHRONIZATION2_NAME,
    vk::KHR_CREATE_RENDERPASS2_NAME,
];

const MAX_FRAMES_IN_FLIGHT: u32 = 2;

const VALIDATION_LAYERS: [&str; 1] = ["VK_LAYER_KHRONOS_validation"];
#[cfg(debug_assertions)]
const VALIDATION_LAYERS_ENABLED: bool = true;
#[cfg(not(debug_assertions))]
const VALIDATION_LAYERS_ENABLED: bool = false;

struct UniformBufferObject {
    model: glm::Mat4,
    view: glm::Mat4,
    proj: glm::Mat4,
}

struct Vertex {
    pos: glm::Vec2,
    color: glm::Vec3,
}

impl Vertex {
    fn get_binding_description() -> vk::VertexInputBindingDescription {
        vk::VertexInputBindingDescription::default()
            .binding(0)
            .stride(size_of::<Vertex>() as u32)
            .input_rate(vk::VertexInputRate::VERTEX)
    }

    fn get_attribute_description() -> [vk::VertexInputAttributeDescription; 2] {
        let pos_attribute_description = vk::VertexInputAttributeDescription::default()
            .binding(0)
            .location(0)
            .format(vk::Format::R32G32_SFLOAT)
            .offset(offset_of!(Vertex, pos) as u32);

        let color_attribute_description = vk::VertexInputAttributeDescription::default()
            .binding(0)
            .location(1)
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(offset_of!(Vertex, color) as u32);

        [pos_attribute_description, color_attribute_description]
    }
}

const VERTICES: [Vertex; 4] = [
    Vertex {
        pos: glm::Vec2 { x: -0.5, y: -0.5 },
        color: glm::Vec3 {
            x: 1.0,
            y: 0.0,
            z: 0.0,
        },
    },
    Vertex {
        pos: glm::Vec2 { x: 0.5, y: -0.5 },
        color: glm::Vec3 {
            x: 0.0,
            y: 1.0,
            z: 0.0,
        },
    },
    Vertex {
        pos: glm::Vec2 { x: 0.5, y: 0.5 },
        color: glm::Vec3 {
            x: 0.0,
            y: 0.0,
            z: 1.0,
        },
    },
    Vertex {
        pos: glm::Vec2 { x: -0.5, y: 0.5 },
        color: glm::Vec3 {
            x: 1.0,
            y: 1.0,
            z: 1.0,
        },
    },
];

const INDICES: [u16; 6] = [0, 1, 2, 2, 3, 0];

struct VulkanRenderer {
    glfw: glfw::Glfw,
    window: glfw::PWindow,

    // entry: ash::Entry,
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

    descriptor_set_layout: vk::DescriptorSetLayout,

    pipeline_layout: vk::PipelineLayout,
    graphics_pipeline: vk::Pipeline,

    command_pool: vk::CommandPool,
    command_buffers: Vec<vk::CommandBuffer>,

    present_complete_semaphores: Vec<vk::Semaphore>,
    render_finished_semaphores: Vec<vk::Semaphore>,
    in_flight_fences: Vec<vk::Fence>,

    frame_index: u32,

    vertex_buffer: vk::Buffer,
    vertex_buffer_memory: vk::DeviceMemory,

    index_buffer: vk::Buffer,
    index_buffer_memory: vk::DeviceMemory,

    uniform_buffers: Vec<vk::Buffer>,
    uniform_buffers_memory: Vec<vk::DeviceMemory>,
    uniform_buffers_mapped: Vec<*mut c_void>,

    start_time: Instant,
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

        let descriptor_set_layout = VulkanRenderer::create_descriptor_set_layout(&device);

        let (pipeline_layout, graphics_pipeline) = VulkanRenderer::create_graphics_pipeline(
            &device,
            &swapchain_image_format,
            &descriptor_set_layout,
        );

        let command_pool = VulkanRenderer::create_command_pool(graphics_family, &device);

        let (vertex_buffer, vertex_buffer_memory) = VulkanRenderer::create_vertex_buffer(
            &instance,
            &physical_device,
            &device,
            &command_pool,
            &graphics_queue,
        );

        let (index_buffer, index_buffer_memory) = VulkanRenderer::create_index_buffer(
            &instance,
            &physical_device,
            &device,
            &command_pool,
            &graphics_queue,
        );

        let (uniform_buffers, uniform_buffers_memory, uniform_buffers_mapped) =
            VulkanRenderer::create_uniform_buffers(&instance, &physical_device, &device);

        let command_buffers = VulkanRenderer::create_command_buffers(&command_pool, &device);

        let (present_complete_semaphores, render_finished_semaphores, in_flight_fences) =
            VulkanRenderer::create_sync_objects(&device, swapchain_images.len());

        let start_time = Instant::now();

        let frame_index = 0;
        VulkanRenderer {
            glfw,
            window,
            // entry,
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
            descriptor_set_layout,
            pipeline_layout,
            graphics_pipeline,
            command_pool,
            command_buffers,
            present_complete_semaphores,
            render_finished_semaphores,
            in_flight_fences,
            frame_index,
            vertex_buffer,
            vertex_buffer_memory,
            index_buffer,
            index_buffer_memory,
            uniform_buffers,
            uniform_buffers_memory,
            uniform_buffers_mapped,
            start_time,
        }
    }

    pub fn run(&mut self) {
        self.main_loop();
    }

    fn init_glfw_window() -> (glfw::Glfw, glfw::PWindow) {
        let mut glfw = glfw::init(glfw::fail_on_errors).unwrap();
        glfw.window_hint(glfw::WindowHint::ClientApi(glfw::ClientApiHint::NoApi));
        glfw.window_hint(glfw::WindowHint::Resizable(true));
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
            if !extension_properties.contains(extension) {
                panic!("Required extension not supported! {extension}")
            }
        }

        // Converting extensions to format applicable to Create info struct
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
        let create_info = vk::InstanceCreateInfo::default()
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
            (None, None)
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
                .pfn_user_callback(Some(VulkanRenderer::debug_callback));

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
        let device_extensions = Vec::from(DEVICE_EXTENSIONS);

        let mut physical_device: Option<vk::PhysicalDevice> = None;
        // Finding suitable devices
        devices.iter().find(|device| {
            let queue_families =
                unsafe { instance.get_physical_device_queue_family_properties(**device) };

            // We want Vulkan version support at least 1.3
            let supports_vulkan_1_3 = unsafe {
                instance
                    .get_physical_device_properties(**device)
                    .api_version
            } >= vk::API_VERSION_1_3;

            // We want access to a graphics queue
            let qfp_iter = queue_families
                .iter()
                .find(|qfp| qfp.queue_flags & vk::QueueFlags::GRAPHICS != vk::QueueFlags::empty());
            let supports_graphics = qfp_iter.is_some();

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
            let supports_all_required_extensions = found;

            let mut vulkan13_features = vk::PhysicalDeviceVulkan13Features::default();
            let mut extended_features =
                vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT::default();
            let mut vulkan11_features = vk::PhysicalDeviceVulkan11Features::default();
            let mut features = vk::PhysicalDeviceFeatures2::default()
                .push_next(&mut vulkan13_features)
                .push_next(&mut vulkan11_features)
                .push_next(&mut extended_features);
            unsafe { instance.get_physical_device_features2(**device, &mut features) };

            let supports_required_features = vulkan11_features.shader_draw_parameters == vk::TRUE
                && vulkan13_features.dynamic_rendering == vk::TRUE
                && extended_features.extended_dynamic_state == vk::TRUE
                && vulkan13_features.synchronization2 == vk::TRUE;

            if supports_vulkan_1_3
                && supports_graphics
                && supports_all_required_extensions
                && supports_required_features
            {
                physical_device = Some(**device);
                return true;
            }
            false
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
        let mut vulkan13_features = vk::PhysicalDeviceVulkan13Features::default()
            .dynamic_rendering(true)
            .synchronization2(true);
        let mut extended_features = vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT::default()
            .extended_dynamic_state(true);
        let mut vulkan11_features =
            vk::PhysicalDeviceVulkan11Features::default().shader_draw_parameters(true);
        device_features = device_features
            .push_next(&mut vulkan13_features)
            .push_next(&mut extended_features)
            .push_next(&mut vulkan11_features);

        let device_extensions = Vec::from(DEVICE_EXTENSIONS)
            .iter()
            .map(|ex| ex.as_ptr())
            .collect::<Vec<_>>();

        let queue_create_infos = [device_queue_create_info];
        let device_create_info = vk::DeviceCreateInfo::default()
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
            .first()
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
        let image_view_create_info = vk::ImageViewCreateInfo::default()
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

    fn read_shader_file(filepath: &str) -> Vec<u32> {
        let raw_file = fs::read(filepath).expect("Failed to read shader file");
        let mut shader_code: Vec<u32> = Vec::new();

        if !raw_file.len().is_multiple_of(4) {
            panic!("Shader file isn't multiple of 4 bytes long, can't parse it as a SPIR-V shader");
        }

        let mut i = 0;
        while i < raw_file.len() - 1 {
            let mut dword = raw_file[i] as u32;
            for _ in 1..4 {
                i += 1;
                dword = (dword << 8) + raw_file[i] as u32;
            }
            shader_code.push(dword);
            i += 1;
        }
        shader_code.iter().map(|c| c.to_be()).collect::<Vec<_>>()
    }

    fn create_descriptor_set_layout(device: &ash::Device) -> vk::DescriptorSetLayout {
        let ubo_layout_binding = vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::VERTEX);

        let bindings = [ubo_layout_binding];
        let layout_create_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);

        unsafe { device.create_descriptor_set_layout(&layout_create_info, None) }
            .expect("Failed to create a descriptor set layout")
    }

    fn create_graphics_pipeline(
        device: &ash::Device,
        swapchain_image_format: &vk::Format,
        descriptor_set_layout: &vk::DescriptorSetLayout,
    ) -> (vk::PipelineLayout, vk::Pipeline) {
        let shader_module = VulkanRenderer::create_shader_module(
            &VulkanRenderer::read_shader_file("shaders/shader.spv"),
            device,
        );

        let vert_shader_name = CString::new("vertMain").expect("Failed to create a CString");
        let vert_shader_stage_info = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(shader_module)
            .name(vert_shader_name.as_c_str());

        let frag_shader_name = CString::new("fragMain").expect("Failed to create a CString");
        let frag_shader_stage_info = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(shader_module)
            .name(frag_shader_name.as_c_str());

        let shader_stages = [vert_shader_stage_info, frag_shader_stage_info];

        let binding_descriptions = [Vertex::get_binding_description()];
        let attribute_descriptions = Vertex::get_attribute_description();

        let vertex_input_info = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(&binding_descriptions)
            .vertex_attribute_descriptions(&attribute_descriptions);

        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST);

        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];

        let dynamic_state_create_info =
            vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);

        let rasterizer = vk::PipelineRasterizationStateCreateInfo::default()
            .depth_clamp_enable(false)
            .rasterizer_discard_enable(false)
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(vk::CullModeFlags::BACK)
            .front_face(vk::FrontFace::CLOCKWISE)
            .depth_bias_enable(false)
            .depth_bias_slope_factor(1f32)
            .line_width(1f32);

        let multisampling = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1)
            .sample_shading_enable(false);

        let color_blend_attachment = vk::PipelineColorBlendAttachmentState::default()
            .blend_enable(true)
            .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
            .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .color_blend_op(vk::BlendOp::ADD)
            .src_alpha_blend_factor(vk::BlendFactor::ONE)
            .dst_alpha_blend_factor(vk::BlendFactor::ZERO)
            .color_write_mask(vk::ColorComponentFlags::RGBA);

        let color_blend_attachments = [color_blend_attachment];
        let color_blend_create_info = vk::PipelineColorBlendStateCreateInfo::default()
            .logic_op_enable(false)
            .logic_op(vk::LogicOp::COPY)
            .attachments(&color_blend_attachments);

        let descriptor_set_layouts = [*descriptor_set_layout];
        let pipeline_layout_info =
            vk::PipelineLayoutCreateInfo::default().set_layouts(&descriptor_set_layouts);

        let pipeline_layout = unsafe { device.create_pipeline_layout(&pipeline_layout_info, None) }
            .expect("Failed to create a pipeline layout");

        let color_attachment_formats = [*swapchain_image_format];
        let mut pipeline_rendering_create_info = vk::PipelineRenderingCreateInfo::default()
            .color_attachment_formats(&color_attachment_formats);

        let pipeline_create_info = vk::GraphicsPipelineCreateInfo::default()
            .push_next(&mut pipeline_rendering_create_info)
            .stages(&shader_stages)
            .vertex_input_state(&vertex_input_info)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterizer)
            .multisample_state(&multisampling)
            .color_blend_state(&color_blend_create_info)
            .dynamic_state(&dynamic_state_create_info)
            .layout(pipeline_layout);

        let pipeline_infos = [pipeline_create_info];
        let pipelines = unsafe {
            device.create_graphics_pipelines(vk::PipelineCache::null(), &pipeline_infos, None)
        }
        .expect("Failed to create graphics pipelines");

        let pipeline = pipelines.first().expect("Failed to get graphics pipeline");

        unsafe { device.destroy_shader_module(shader_module, None) };

        (pipeline_layout, *pipeline)
    }

    fn create_shader_module(code: &Vec<u32>, device: &ash::Device) -> vk::ShaderModule {
        // let code = code.iter().map(|ch| *ch as u32).collect::<Vec<_>>();
        let shader_module_create_info = vk::ShaderModuleCreateInfo::default().code(code.as_slice());
        unsafe { device.create_shader_module(&shader_module_create_info, None) }
            .expect("Failed to create a shader module")
    }

    fn create_command_pool(graphics_family: u32, device: &ash::Device) -> vk::CommandPool {
        let pool_info = vk::CommandPoolCreateInfo::default()
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            .queue_family_index(graphics_family);

        unsafe { device.create_command_pool(&pool_info, None) }
            .expect("Failed to create a command pool")
    }

    fn create_buffer(
        instance: &ash::Instance,
        physical_device: &vk::PhysicalDevice,
        device: &ash::Device,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        properties: vk::MemoryPropertyFlags,
    ) -> (vk::Buffer, vk::DeviceMemory) {
        let buffer_create_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let buffer = unsafe { device.create_buffer(&buffer_create_info, None) }
            .expect("Failed to create a buffer");

        let memory_requirements = unsafe { device.get_buffer_memory_requirements(buffer) };

        let memory_alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(memory_requirements.size)
            .memory_type_index(VulkanRenderer::find_memory_type(
                instance,
                physical_device,
                memory_requirements.memory_type_bits,
                properties,
            ));

        let buffer_memory = unsafe { device.allocate_memory(&memory_alloc_info, None) }
            .expect("Failed to allocate memory for buffer");

        unsafe { device.bind_buffer_memory(buffer, buffer_memory, 0) }
            .expect("Failed to bind buffer");

        (buffer, buffer_memory)
    }

    fn create_vertex_buffer(
        instance: &ash::Instance,
        physical_device: &vk::PhysicalDevice,
        device: &ash::Device,
        command_pool: &vk::CommandPool,
        graphics_queue: &vk::Queue,
    ) -> (vk::Buffer, vk::DeviceMemory) {
        let vertex_buffer_size: vk::DeviceSize = (size_of::<Vertex>() * VERTICES.len()) as u64;

        let (staging_buffer, staging_buffer_memory) = VulkanRenderer::create_buffer(
            instance,
            physical_device,
            device,
            vertex_buffer_size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        );

        let data_ptr = unsafe {
            device.map_memory(
                staging_buffer_memory,
                0,
                vertex_buffer_size,
                vk::MemoryMapFlags::empty(),
            )
        }
        .expect("Failed to map memory");

        unsafe {
            data_ptr.copy_from_nonoverlapping(
                VERTICES.as_ptr() as *const c_void,
                vertex_buffer_size as usize,
            )
        };

        unsafe { device.unmap_memory(staging_buffer_memory) };

        let (vertex_buffer, vertex_buffer_memory) = VulkanRenderer::create_buffer(
            instance,
            physical_device,
            device,
            vertex_buffer_size,
            vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        );

        VulkanRenderer::copy_buffer(
            command_pool,
            device,
            graphics_queue,
            &staging_buffer,
            &vertex_buffer,
            vertex_buffer_size,
        );

        unsafe { device.free_memory(staging_buffer_memory, None) };
        unsafe { device.destroy_buffer(staging_buffer, None) };

        (vertex_buffer, vertex_buffer_memory)
    }

    fn create_index_buffer(
        instance: &ash::Instance,
        physical_device: &vk::PhysicalDevice,
        device: &ash::Device,
        command_pool: &vk::CommandPool,
        graphics_queue: &vk::Queue,
    ) -> (vk::Buffer, vk::DeviceMemory) {
        let index_buffer_size: vk::DeviceSize = (size_of::<u16>() * INDICES.len()) as u64;

        let (staging_buffer, staging_buffer_memory) = VulkanRenderer::create_buffer(
            instance,
            physical_device,
            device,
            index_buffer_size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        );

        let data_ptr = unsafe {
            device.map_memory(
                staging_buffer_memory,
                0,
                index_buffer_size,
                vk::MemoryMapFlags::empty(),
            )
        }
        .expect("Failed to map memory");

        unsafe {
            data_ptr.copy_from_nonoverlapping(
                INDICES.as_ptr() as *const c_void,
                index_buffer_size as usize,
            )
        };

        unsafe { device.unmap_memory(staging_buffer_memory) };

        let (index_buffer, index_buffer_memory) = VulkanRenderer::create_buffer(
            instance,
            physical_device,
            device,
            index_buffer_size,
            vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        );

        VulkanRenderer::copy_buffer(
            command_pool,
            device,
            graphics_queue,
            &staging_buffer,
            &index_buffer,
            index_buffer_size,
        );

        unsafe { device.free_memory(staging_buffer_memory, None) };
        unsafe { device.destroy_buffer(staging_buffer, None) };

        (index_buffer, index_buffer_memory)
    }

    fn create_uniform_buffers(
        instance: &ash::Instance,
        physical_device: &vk::PhysicalDevice,
        device: &ash::Device,
    ) -> (Vec<vk::Buffer>, Vec<vk::DeviceMemory>, Vec<*mut c_void>) {
        let (mut buffers, mut buffers_memory, mut buffers_mapped) =
            (Vec::new(), Vec::new(), Vec::new());

        let buffer_size = size_of::<UniformBufferObject>() as u64;
        for _ in 0..MAX_FRAMES_IN_FLIGHT {
            let (buffer, buffer_memory) = VulkanRenderer::create_buffer(
                instance,
                physical_device,
                device,
                buffer_size,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            );
            buffers.push(buffer);
            buffers_memory.push(buffer_memory);

            let data_ptr = unsafe {
                device.map_memory(buffer_memory, 0, buffer_size, vk::MemoryMapFlags::empty())
            }
            .expect("Failed to map memory");

            buffers_mapped.push(data_ptr);
        }

        (buffers, buffers_memory, buffers_mapped)
    }

    fn copy_buffer(
        command_pool: &vk::CommandPool,
        device: &ash::Device,
        graphics_queue: &vk::Queue,
        src_buffer: &vk::Buffer,
        dst_buffer: &vk::Buffer,
        size: vk::DeviceSize,
    ) {
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(*command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);

        let command_buffers = unsafe { device.allocate_command_buffers(&alloc_info) }
            .expect("Failed to allocate copy command buffer");

        let command_copy_buffer = command_buffers
            .first()
            .expect("Allocated 0 command buffers");

        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe { device.begin_command_buffer(*command_copy_buffer, &begin_info) }
            .expect("Failed to begin copy command buffer");
        unsafe {
            device.cmd_copy_buffer(
                *command_copy_buffer,
                *src_buffer,
                *dst_buffer,
                &[vk::BufferCopy::default()
                    .size(size)
                    .src_offset(0)
                    .dst_offset(0)],
            )
        };
        unsafe { device.end_command_buffer(*command_copy_buffer) }
            .expect("Failed to end copy command buffer");

        let command_buffers = [*command_copy_buffer];
        let queue_submit_info = vk::SubmitInfo::default().command_buffers(&command_buffers);
        unsafe { device.queue_submit(*graphics_queue, &[queue_submit_info], vk::Fence::null()) }
            .expect("Failed to copy queue submit");
        unsafe { device.queue_wait_idle(*graphics_queue) }.expect("Failed to wait on copy queue");
    }

    fn find_memory_type(
        instance: &ash::Instance,
        physical_device: &vk::PhysicalDevice,
        type_filter: u32,
        properties: vk::MemoryPropertyFlags,
    ) -> u32 {
        let memory_properties =
            unsafe { instance.get_physical_device_memory_properties(*physical_device) };

        for (idx, mem_type) in memory_properties.memory_types_as_slice().iter().enumerate() {
            if (type_filter & (1 << idx) != 0)
                && (mem_type.property_flags & properties == properties)
            {
                return idx as u32;
            }
        }

        panic!("Failed to find a suitable memory type");
    }

    fn create_command_buffers(
        command_pool: &vk::CommandPool,
        device: &ash::Device,
    ) -> Vec<vk::CommandBuffer> {
        let allocate_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(*command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(MAX_FRAMES_IN_FLIGHT);

        unsafe {
            device
                .allocate_command_buffers(&allocate_info)
                .expect("Failed to allocate command buffers")
        }
    }

    fn create_sync_objects(
        device: &ash::Device,
        swapchain_images_count: usize,
    ) -> (Vec<vk::Semaphore>, Vec<vk::Semaphore>, Vec<vk::Fence>) {
        let mut present_complete_semaphores: Vec<vk::Semaphore> = Vec::new();
        for _ in 0..MAX_FRAMES_IN_FLIGHT {
            let present_complete_semaphore =
                unsafe { device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None) }
                    .expect("Failed to create a semaphore");
            present_complete_semaphores.push(present_complete_semaphore);
        }

        let mut render_finished_semaphores: Vec<vk::Semaphore> = Vec::new();
        for _ in 0..swapchain_images_count {
            let render_finished_semaphore =
                unsafe { device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None) }
                    .expect("Failed to create a semaphore");
            render_finished_semaphores.push(render_finished_semaphore);
        }

        let mut in_flight_fences: Vec<vk::Fence> = Vec::new();
        for _ in 0..MAX_FRAMES_IN_FLIGHT {
            let in_flight_fence = unsafe {
                device.create_fence(
                    &vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED),
                    None,
                )
            }
            .expect("Failed to create a fence");
            in_flight_fences.push(in_flight_fence);
        }

        (
            present_complete_semaphores,
            render_finished_semaphores,
            in_flight_fences,
        )
    }

    fn recreate_swapchain(&mut self) {
        let (mut width, mut height) = self.window.get_framebuffer_size();
        while width == 0 || height == 0 {
            (width, height) = self.window.get_framebuffer_size();
            self.glfw.wait_events();
        }

        unsafe {
            self.device
                .device_wait_idle()
                .expect("Failed to wait for idle device")
        };

        self.cleanup_swapchain();

        let swapchain_stuff = VulkanRenderer::create_swapchain(
            &self.surface_instance,
            &self.surface,
            &self.physical_device,
            &self.instance,
            &self.device,
            &self.window,
            self.graphics_family,
            self.presentation_family,
        );

        self.swapchain_device = swapchain_stuff.0;
        self.swapchain = swapchain_stuff.1;
        self.swapchain_images = swapchain_stuff.2;
        self.swapchain_image_format = swapchain_stuff.3;
        self.swapchain_extent = swapchain_stuff.4;

        self.swapchain_image_views = VulkanRenderer::create_image_views(
            self.swapchain_image_format,
            &self.swapchain_images,
            &self.device,
        );
    }

    fn cleanup_swapchain(&mut self) {
        for image_view in &self.swapchain_image_views {
            unsafe { self.device.destroy_image_view(*image_view, None) };
        }
        unsafe {
            self.swapchain_device
                .destroy_swapchain(self.swapchain, None)
        };
    }

    fn record_command_buffer(&self, image_index: usize) {
        unsafe {
            self.device.begin_command_buffer(
                self.command_buffers[self.frame_index as usize],
                &vk::CommandBufferBeginInfo::default(),
            )
        }
        .expect("Failed to begin command buffer");

        self.transition_image_layout(
            image_index,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            vk::AccessFlags2::empty(),
            vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
        );

        let mut clear_color = vk::ClearValue::default();
        clear_color.color = vk::ClearColorValue {
            float32: [0f32, 0f32, 0f32, 1f32],
        };

        let attachment_info = vk::RenderingAttachmentInfo::default()
            .image_view(self.swapchain_image_views[image_index])
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .clear_value(clear_color);

        let color_attachments = [attachment_info];
        let rendering_info = vk::RenderingInfo::default()
            .render_area(
                vk::Rect2D::default()
                    .offset(vk::Offset2D::default().x(0).y(0))
                    .extent(self.swapchain_extent),
            )
            .layer_count(1)
            .color_attachments(&color_attachments);

        unsafe {
            self.device.cmd_begin_rendering(
                self.command_buffers[self.frame_index as usize],
                &rendering_info,
            )
        };

        unsafe {
            self.device.cmd_bind_pipeline(
                self.command_buffers[self.frame_index as usize],
                vk::PipelineBindPoint::GRAPHICS,
                self.graphics_pipeline,
            )
        };

        unsafe {
            self.device.cmd_bind_vertex_buffers(
                self.command_buffers[self.frame_index as usize],
                0,
                &[self.vertex_buffer],
                &[0],
            )
        };
        unsafe {
            self.device.cmd_bind_index_buffer(
                self.command_buffers[self.frame_index as usize],
                self.index_buffer,
                0,
                vk::IndexType::UINT16,
            )
        };

        let viewports = [vk::Viewport::default()
            .x(0f32)
            .y(0f32)
            .width(self.swapchain_extent.width as f32)
            .height(self.swapchain_extent.height as f32)
            .min_depth(0f32)
            .max_depth(1f32)];
        unsafe {
            self.device.cmd_set_viewport(
                self.command_buffers[self.frame_index as usize],
                0,
                &viewports,
            )
        };

        let scissors = [vk::Rect2D::default()
            .offset(vk::Offset2D::default().x(0).y(0))
            .extent(self.swapchain_extent)];
        unsafe {
            self.device.cmd_set_scissor(
                self.command_buffers[self.frame_index as usize],
                0,
                &scissors,
            )
        };

        unsafe {
            self.device.cmd_draw_indexed(
                self.command_buffers[self.frame_index as usize],
                INDICES.len() as u32,
                1,
                0,
                0,
                0,
            )
        };

        unsafe {
            self.device
                .cmd_end_rendering(self.command_buffers[self.frame_index as usize])
        };

        self.transition_image_layout(
            image_index,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            vk::ImageLayout::PRESENT_SRC_KHR,
            vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            vk::AccessFlags2::empty(),
            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            vk::PipelineStageFlags2::BOTTOM_OF_PIPE,
        );

        unsafe {
            self.device
                .end_command_buffer(self.command_buffers[self.frame_index as usize])
        }
        .expect("Failed to end command buffer");
    }

    fn transition_image_layout(
        &self,
        image_index: usize,
        old_layout: vk::ImageLayout,
        new_layout: vk::ImageLayout,
        src_access_mask: vk::AccessFlags2,
        dst_access_mask: vk::AccessFlags2,
        src_stage_mask: vk::PipelineStageFlags2,
        dst_stage_mask: vk::PipelineStageFlags2,
    ) {
        let barrier = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(src_stage_mask)
            .src_access_mask(src_access_mask)
            .dst_stage_mask(dst_stage_mask)
            .dst_access_mask(dst_access_mask)
            .old_layout(old_layout)
            .new_layout(new_layout)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(self.swapchain_images[image_index])
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_mip_level(0)
                    .level_count(1)
                    .base_array_layer(0)
                    .layer_count(1),
            );

        let barriers = [barrier];
        let dependency_info = vk::DependencyInfo::default().image_memory_barriers(&barriers);
        unsafe {
            self.device.cmd_pipeline_barrier2(
                self.command_buffers[self.frame_index as usize],
                &dependency_info,
            )
        };
    }

    fn main_loop(&mut self) {
        while !self.window.should_close() {
            self.glfw.poll_events();
            self.draw_frame();
        }
        unsafe {
            self.device
                .device_wait_idle()
                .expect("Failed to wait for device idle")
        };
    }

    fn draw_frame(&mut self) {
        let fence_result = unsafe {
            self.device.wait_for_fences(
                &[self.in_flight_fences[self.frame_index as usize]],
                true,
                u64::MAX,
            )
        };

        fence_result.expect("Failed to wait for fence");

        let acquire_next_image_result = unsafe {
            self.swapchain_device.acquire_next_image(
                self.swapchain,
                u64::MAX,
                self.present_complete_semaphores[self.frame_index as usize],
                vk::Fence::null(),
            )
        };

        let image_index: u32 = match acquire_next_image_result {
            Ok((idx, _)) => idx,
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                self.recreate_swapchain();
                return;
            }
            _ => panic!("Failed to acquire swapchain image"),
        };

        unsafe {
            self.device
                .reset_fences(&[self.in_flight_fences[self.frame_index as usize]])
        }
        .expect("Failed to reset fences");

        unsafe {
            self.device.reset_command_buffer(
                self.command_buffers[self.frame_index as usize],
                vk::CommandBufferResetFlags::empty(),
            )
        }
        .expect("Failed to reset command buffer");
        self.record_command_buffer(image_index as usize);

        self.update_uniform_buffer(self.frame_index);

        let wait_dst_stage_mask = vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT;
        let wait_semaphores = [self.present_complete_semaphores[self.frame_index as usize]];
        let wait_dst_stage_masks = [wait_dst_stage_mask];
        let command_buffers = [self.command_buffers[self.frame_index as usize]];
        let signal_semaphores = [self.render_finished_semaphores[image_index as usize]];
        let submit_info = vk::SubmitInfo::default()
            .wait_semaphores(&wait_semaphores)
            .wait_dst_stage_mask(&wait_dst_stage_masks)
            .command_buffers(&command_buffers)
            .signal_semaphores(&signal_semaphores);

        let submits = [submit_info];
        unsafe {
            self.device.queue_submit(
                self.graphics_queue,
                &submits,
                self.in_flight_fences[self.frame_index as usize],
            )
        }
        .expect("Failed to submit commands to queue");

        let wait_semaphores = [self.render_finished_semaphores[image_index as usize]];
        let swapchains = [self.swapchain];
        let image_indices = [image_index];
        let present_info_khr = vk::PresentInfoKHR::default()
            .wait_semaphores(&wait_semaphores)
            .swapchains(&swapchains)
            .image_indices(&image_indices);

        let queue_present_result = unsafe {
            self.swapchain_device
                .queue_present(self.presentation_queue, &present_info_khr)
        };

        match queue_present_result {
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) | Ok(true) => self.recreate_swapchain(),
            Ok(false) => {}
            _ => panic!("Failed to present"),
        }

        self.frame_index = (self.frame_index + 1) % MAX_FRAMES_IN_FLIGHT
    }

    fn update_uniform_buffer(&self, frame_index: u32) {
        let current_time = Instant::now();
        let time = current_time.duration_since(self.start_time).as_secs_f32();

        let mut ubo = UniformBufferObject {
            model: glm::ext::rotate(
                &glm::mat4(
                    1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                ),
                time * glm::radians(90.0),
                glm::vec3(0.0, 0.0, 1.0),
            ),
            view: glm::ext::look_at(
                glm::vec3(2.0, 2.0, 2.0),
                glm::vec3(0.0, 0.0, 0.0),
                glm::vec3(0.0, 0.0, 1.0),
            ),
            proj: glm::ext::perspective(
                glm::radians(45.0),
                self.swapchain_extent.width as f32 / self.swapchain_extent.height as f32,
                0.1,
                1.0,
            ),
        };

        ubo.proj[1][1] *= -1.0;

        unsafe {
            self.uniform_buffers_mapped
                .get(frame_index as usize)
                .expect("Failed to get uniform buffer memory")
                .copy_from_nonoverlapping(
                    &mut ubo as *mut _ as *const c_void,
                    size_of::<UniformBufferObject>(),
                )
        };
    }

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
}

impl Drop for VulkanRenderer {
    // Cleanup code
    fn drop(&mut self) {
        unsafe {
            for in_flight_fence in &self.in_flight_fences {
                self.device.destroy_fence(*in_flight_fence, None);
            }
            for present_complete_semaphore in &self.present_complete_semaphores {
                self.device
                    .destroy_semaphore(*present_complete_semaphore, None);
            }
            for render_finished_semaphore in &self.render_finished_semaphores {
                self.device
                    .destroy_semaphore(*render_finished_semaphore, None);
            }
            for buffer_memory in &self.uniform_buffers_memory {
                self.device.free_memory(*buffer_memory, None);
            }
            for buffer in &self.uniform_buffers {
                self.device.destroy_buffer(*buffer, None);
            }
            self.device.free_memory(self.index_buffer_memory, None);
            self.device.destroy_buffer(self.index_buffer, None);
            self.device.free_memory(self.vertex_buffer_memory, None);
            self.device.destroy_buffer(self.vertex_buffer, None);
            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_pipeline(self.graphics_pipeline, None);
            self.device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.device
                .destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            self.cleanup_swapchain();
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
