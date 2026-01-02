use ash::{Entry, vk};
use glfw;

fn main() {
    let mut app = HelloTriangleApplication::new();

    app.run();
}

const WIDTH: u32 = 800;
const HEIGHT: u32 = 600;

struct HelloTriangleApplication {
    instance: ash::Instance,
    glfw: glfw::Glfw,
    window: glfw::PWindow,
}

impl HelloTriangleApplication {
    // Initialization code
    pub fn new() -> Self {
        // init_window()
        let mut glfw = glfw::init(glfw::fail_on_errors).unwrap();
        let (window, _) = glfw
            .create_window(WIDTH, HEIGHT, "Vulkan", glfw::WindowMode::Windowed)
            .expect("Failed to create GLFW window.");

        // init_vulkan()
        let entry = Entry::linked();

        let app_info = vk::ApplicationInfo {
            p_application_name: "Hello Triangle".as_ptr() as *const i8,
            application_version: vk::make_api_version(0, 1, 0, 0),
            p_engine_name: "No Engine".as_ptr() as *const i8,
            engine_version: vk::make_api_version(0, 1, 0, 0),
            api_version: vk::API_VERSION_1_0,
            ..Default::default()
        };

        let create_info = vk::InstanceCreateInfo {
            p_application_info: &app_info,
            ..Default::default()
        };

        let instance = unsafe {
            entry
                .create_instance(&create_info, None)
                .expect("Failed to create a Vulkan Instance")
        };

        HelloTriangleApplication {
            instance,
            glfw,
            window,
        }
    }

    pub fn run(&mut self) {
        self.main_loop();
    }

    fn main_loop(&mut self) {
        while !self.window.should_close() {
            self.glfw.poll_events();
        }
    }
}

impl Drop for HelloTriangleApplication {
    // Cleanup code
    fn drop(&mut self) {}
}
