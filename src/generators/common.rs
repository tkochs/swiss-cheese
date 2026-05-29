pub mod mode {
    pub enum Mode {
        GM,
        MAX,
        MIN,
    }
    pub const ALLOWED_MODES: &[&str] = &["GM", "MAX", "MIN"];

    impl Into<Mode> for &str {
        fn into(self) -> Mode {
            match self.to_lowercase().as_str() {
                "gm" => Mode::GM,
                "max" => Mode::MAX,
                "min" => Mode::MIN,
                _ => {
                    panic!(
                        "Unknown mode parameter: {} (Alloed: {:?})",
                        self, ALLOWED_MODES
                    )
                }
            }
        }
    }
}
