use ash::{Entry, vk};
use glfw::{self};

fn main() {
    let mut app = VulkanRenderer::new();

    app.run();
}

const WIDTH: u32 = 800;
const HEIGHT: u32 = 600;

struct VulkanRenderer {
    instance: ash::Instance,
    glfw: glfw::Glfw,
    window: glfw::PWindow,
}

impl VulkanRenderer {
    // Initialization code
    pub fn new() -> Self {
        let (glfw, window) = VulkanRenderer::init_window();
        let instance = VulkanRenderer::create_vk_instance(&glfw);
        VulkanRenderer {
            instance,
            glfw,
            window,
        }
    }

    pub fn run(&mut self) {
        self.main_loop();
    }

    fn init_window() -> (glfw::Glfw, glfw::PWindow) {
        let mut glfw = glfw::init(glfw::fail_on_errors).unwrap();
        let (window, _) = glfw
            .create_window(WIDTH, HEIGHT, "Vulkan", glfw::WindowMode::Windowed)
            .expect("Failed to create GLFW window");
        (glfw, window)
    }

    fn create_vk_instance(glfw: &glfw::Glfw) -> ash::Instance {
        let entry = Entry::linked();

        let app_info = vk::ApplicationInfo {
            p_application_name: "Hello Triangle".as_ptr() as *const i8,
            application_version: vk::make_api_version(0, 1, 0, 0),
            p_engine_name: "No Engine".as_ptr() as *const i8,
            engine_version: vk::make_api_version(0, 1, 0, 0),
            api_version: vk::API_VERSION_1_3,
            ..Default::default()
        };

        let glfw_extensions = glfw
            .get_required_instance_extensions()
            .expect("Failed to get required GLFW extensions");

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

        //?[DEBUG]
        println!("Available extensions:");
        for name in &extension_properties {
            println!("{name}")
        }
        println!("\nGLFW required extensions:");
        for name in &glfw_extensions {
            println!("{name}")
        }

        for extension in &glfw_extensions {
            if !extension_properties.contains(&extension) {
                panic!("Required GLFW extension not supported! {extension}")
            }
        }

        let extensions_count = glfw_extensions.len() as u32;

        let extensions = glfw_extensions
            .iter()
            .map(|name| name.as_ptr() as *const i8)
            .collect::<Vec<_>>();
        let extensions = extensions.as_ptr();

        let create_info = vk::InstanceCreateInfo {
            p_application_info: &app_info,
            enabled_extension_count: extensions_count,
            pp_enabled_extension_names: extensions,
            ..Default::default()
        };

        unsafe {
            entry
                .create_instance(&create_info, None)
                .expect("Failed to create a Vulkan Instance")
        }
    }

    fn main_loop(&mut self) {
        while !self.window.should_close() {
            self.glfw.poll_events();
        }
    }
}

impl Drop for VulkanRenderer {
    // Cleanup code
    fn drop(&mut self) {}
}
