struct MyStruct
    x::Int
    y::Float64
end

function MyStruct(x::Int, y::Float64)
    obj = MyStruct()
    obj.x = x
    obj.y = y
    return obj
end



MyStruct(3, 64.2)