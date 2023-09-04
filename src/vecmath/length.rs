pub trait Length<T> {
    fn length_squared(&self) -> T;
    fn length(&self) -> T;
}
