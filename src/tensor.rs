use std::{collections::HashMap, fs::File, io::{BufReader, Read, Seek}};



enum TensorType
{
    Invalid,
    Uint8,
    Int8,
    Uint16,
    Int16,
    Uint32,
    Int32,
    Uint64,
    Int64,

    Float16,
    Float32,
    Float64,
}

impl TensorType
{
    pub fn size(&self) -> usize
    {
        match self
        {
            TensorType::Invalid => 0,
            TensorType::Uint8 => 1,
            TensorType::Int8 => 1,
            TensorType::Uint16 => 2,
            TensorType::Int16 => 2,
            TensorType::Uint32 => 4,
            TensorType::Int32 => 4,
            TensorType::Uint64 => 8,
            TensorType::Int64 => 8,
            TensorType::Float16 => 2,
            TensorType::Float32 => 4,
            TensorType::Float64 => 8,
        }
    }
}

impl From<u8> for TensorType
{
    fn from(value: u8) -> Self
    {
        match value
        {
            0 => TensorType::Invalid,
            1 => TensorType::Uint8,
            2 => TensorType::Int8,
            3 => TensorType::Uint16,
            4 => TensorType::Int16,
            5 => TensorType::Uint32,
            6 => TensorType::Int32,
            7 => TensorType::Uint64,
            8 => TensorType::Int64,
            9 => TensorType::Float16,
            10 => TensorType::Float32,
            11 => TensorType::Float64,
            _ => panic!("Invalid tensor type"),
        }
    }
}

pub struct TensorField
{
    pub d_type: TensorType,
    // Offset in the file
    pub offset: usize,
    // Specifies the rank and size along each dimension
    pub shape: Vec<usize>,
    /// Data for the field.
    pub data: Vec<u8>,
}

/// Tensor definition used for reading tensor files for measured BSDFs.
pub struct Tensor
{
    pub fields: HashMap<String, TensorField>,
    pub filename: String,
    pub size: usize,
}

impl Tensor
{
    pub fn read(filename: &str) -> Result<Tensor, Box<dyn std::error::Error>>
    {
        let file = File::open(filename)?;

        let size = file.metadata()?.len() as usize;

        assert!(size >= 12 + 2 + 4);
        
        let mut header = [0u8; 12];
        let mut version = [0u8; 2];
        let mut n_fields = [0u8; 4];
        
        let mut reader = BufReader::new(file);

        reader.read_exact(&mut header)?;
        reader.read_exact(&mut version)?;
        reader.read_exact(&mut n_fields)?;
        
        // Note: big-endian
        let n_fields = u32::from_be_bytes(n_fields) as usize;

        // Expected header
        let header = String::from_utf8(header.to_vec())?;
        assert!(header == "tensor_file");

        // Known version
        assert!(version[0] == 1 && version[1] == 0);

        let mut fields = HashMap::with_capacity(n_fields);
        
        for _i in 0..n_fields
        {
            let mut d_type = [0u8; 1];
            let mut name_length = [0u8; 2];
            let mut n_dim = [0u8; 2];
            let mut offset = [0u8; 8];

            reader.read_exact(&mut name_length)?;
            let name_length = u16::from_be_bytes(name_length) as usize;
            let name = {
                let mut name = vec![0u8; name_length];
                // TODO Are we expecting \0 terminated strings?
                reader.read_exact(&mut name)?;
                String::from_utf8(name)?
            };

            reader.read_exact(&mut n_dim)?;
            let n_dim = u16::from_be_bytes(n_dim) as usize;
            reader.read_exact(&mut d_type)?;
            let d_type = d_type[0];
            reader.read_exact(&mut offset)?;
            let offset = u64::from_be_bytes(offset) as usize;

            assert!(d_type != TensorType::Invalid as u8);
            assert!(d_type <= TensorType::Float64 as u8);

            let mut shape: Vec<usize> = Vec::with_capacity(n_dim);
            let mut total_size = TensorType::from(d_type).size();

            for j in 0..n_dim 
            {
                let mut size_value = [0u8; 8];
                reader.read_exact(&mut size_value)?;
                let size_value = u64::from_be_bytes(size_value) as usize;
                shape.push(size_value);
                total_size *= shape[j];
            }

            let mut data = vec![0u8; total_size];
            let cur_pos = reader.seek(std::io::SeekFrom::Current(0))?;
            reader.seek(std::io::SeekFrom::Start(offset as u64))?;
            reader.read_exact(&mut data)?;
            reader.seek(std::io::SeekFrom::Start(cur_pos))?;

            fields.insert(
                name,
                TensorField {
                    d_type: TensorType::from(d_type),
                    offset,
                    shape,
                    data
                });
        }

        Result::Ok(Tensor {
            fields,
            filename: filename.to_string(),
            size
        })
    }
}