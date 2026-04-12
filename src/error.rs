use thiserror::Error;

#[derive(Error, Debug)]
pub enum AppError {
    #[error("Vulkan error: {0}")]
    Vulkan(#[from] ash::vk::Result),

    #[error("Unknown error: {0}")]
    Unknown(&'static str),
}

pub type Result<T> = std::result::Result<T, AppError>;
