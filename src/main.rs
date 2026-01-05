#![allow(unused)]
#![allow(dead_code)]
use ash::{
    Entry, ext,
    vk::{self, PhysicalDevice},
};
use glfw::{self};
use std::{
    ffi::{CStr, CString},
    ops::Deref,
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
        "\n\x1b[34m[DEBUG] {severity_color}[{}] {type_color}[{}]\x1b[0m\n{}",
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

    physical_device: vk::PhysicalDevice,
    device: ash::Device,
}

impl VulkanRenderer {
    // Initialization code
    pub fn new() -> Self {
        let (glfw, window) = VulkanRenderer::init_glfw_window();
        let (entry, instance) = VulkanRenderer::create_vk_instance(&glfw);
        let (debug_instance, debug_messenger) =
            VulkanRenderer::setup_debug_messenger(&entry, &instance);
        let physical_device = VulkanRenderer::pick_physical_device(&instance);
        let device = VulkanRenderer::create_logical_device(&instance, &physical_device);

        VulkanRenderer {
            glfw,
            window,
            entry,
            instance,
            debug_instance,
            debug_messenger,
            physical_device,
            device,
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
        let app_info = vk::ApplicationInfo {
            p_application_name: app_name.as_ptr(),
            application_version: vk::make_api_version(0, 1, 0, 0),
            p_engine_name: engine_name.as_ptr(),
            engine_version: vk::make_api_version(0, 1, 0, 0),
            api_version: vk::API_VERSION_1_3,
            ..Default::default()
        };

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
        let mut create_info = vk::InstanceCreateInfo {
            p_application_info: &app_info,
            enabled_layer_count: layers_count,
            pp_enabled_layer_names: layers.as_ptr(),
            enabled_extension_count: extensions_count,
            pp_enabled_extension_names: extensions.as_ptr(),
            ..Default::default()
        };

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
            let debug_utils_create_info = vk::DebugUtilsMessengerCreateInfoEXT {
                message_severity: severity_flags,
                message_type: message_type_flags,
                pfn_user_callback: Some(debug_callback),
                ..Default::default()
            };

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

    fn find_queue_families(instance: &ash::Instance, physical_device: &vk::PhysicalDevice) -> u32 {
        let queue_family_properties =
            unsafe { instance.get_physical_device_queue_family_properties(*physical_device) };

        queue_family_properties
            .iter()
            .position(|qfp| qfp.queue_flags & vk::QueueFlags::GRAPHICS != vk::QueueFlags::empty())
            .expect("Failed to find a graphics queue") as u32
    }

    fn create_logical_device(
        instance: &ash::Instance,
        physical_device: &vk::PhysicalDevice,
    ) -> ash::Device {
        let queue_family_properties =
            unsafe { instance.get_physical_device_queue_family_properties(*physical_device) };
        let graphics_index = VulkanRenderer::find_queue_families(instance, physical_device);
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

        unsafe { instance.create_device(*physical_device, &device_create_info, None) }
            .expect("Failed to create a logical device")
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
            self.device.destroy_device(None);
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
