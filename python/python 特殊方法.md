# Python 常用双下划线方法（特殊方法）对照表

## 1. 对象构造 & 析构
| 方法 | 调用时机 | 示例 |
|------|----------|------|
| `__init__(self, ...)` | 创建对象后初始化 | `obj = MyClass()` |
| `__new__(cls, ...)` | 创建对象前（控制实例化） | 单例模式 |
| `__del__(self)` | 对象被销毁时（不可靠，少用） | `del obj` 或垃圾回收 |

---

## 2. 对象表示
| 方法 | 调用时机 | 示例 |
|------|----------|------|
| `__str__(self)` | `str(obj)` 或 `print(obj)` 时 | 友好可读的字符串 |
| `__repr__(self)` | 解释器显示、`repr(obj)` 时 | 调试/官方字符串表示 |
| `__format__(self, spec)` | `format(obj, spec)` 或 f-string 格式化 | `f"{obj:.2f}"` |
| `__bytes__(self)` | `bytes(obj)` | 转换为字节串 |

---

## 3. 比较 & 哈希
| 方法 | 调用时机 | 示例 |
|------|----------|------|
| `__eq__(self, other)` | `==` | 相等比较 |
| `__ne__(self, other)` | `!=` | 不等比较 |
| `__lt__(self, other)` | `<` | 小于 |
| `__le__(self, other)` | `<=` | 小于等于 |
| `__gt__(self, other)` | `>` | 大于 |
| `__ge__(self, other)` | `>=` | 大于等于 |
| `__hash__(self)` | `hash(obj)`、用作字典键 | 返回整数哈希值 |

---

## 4. 容器协议（索引/迭代）
| 方法 | 调用时机 | 示例 |
|------|----------|------|
| `__len__(self)` | `len(obj)` | 返回长度 |
| `__getitem__(self, key)` | `obj[key]` | 获取元素 |
| `__setitem__(self, key, value)` | `obj[key] = value` | 设置元素 |
| `__delitem__(self, key)` | `del obj[key]` | 删除元素 |
| `__iter__(self)` | `for x in obj` | 返回迭代器 |
| `__next__(self)` | `next(iterator)` | 迭代器取下一个值 |
| `__contains__(self, item)` | `item in obj` | 成员检测 |

---

## 5. 可调用对象
| 方法 | 调用时机 | 示例 |
|------|----------|------|
| `__call__(self, *args, **kwargs)` | `obj(...)` | 像函数一样调用对象 |

---

## 6. 数值运算
| 方法 | 调用时机 | 示例 |
|------|----------|------|
| `__add__(self, other)` | `+` | 加法 |
| `__sub__(self, other)` | `-` | 减法 |
| `__mul__(self, other)` | `*` | 乘法 |
| `__matmul__(self, other)` | `@` | 矩阵乘法 |
| `__truediv__(self, other)` | `/` | 真除法 |
| `__floordiv__(self, other)` | `//` | 地板除法 |
| `__mod__(self, other)` | `%` | 取模 |
| `__pow__(self, other)` | `**` | 幂运算 |
| `__neg__(self)` | `-obj` | 取负 |
| `__abs__(self)` | `abs(obj)` | 绝对值 |

---

## 7. 上下文管理
| 方法 | 调用时机 | 示例 |
|------|----------|------|
| `__enter__(self)` | `with` 语句开始 | `with obj as x:` |
| `__exit__(self, exc_type, exc_value, traceback)` | `with` 语句结束 | 释放资源/异常处理 |

---

## 8. 反射 & 属性访问
| 方法 | 调用时机 | 示例 |
|------|----------|------|
| `__getattr__(self, name)` | 访问不存在属性时 | `obj.foo` |
| `__getattribute__(self, name)` | 访问任何属性时（都会调用） | 慎用 |
| `__setattr__(self, name, value)` | 设置属性时 | `obj.foo = 1` |
| `__delattr__(self, name)` | 删除属性时 | `del obj.foo` |
| `__dir__(self)` | `dir(obj)` | 返回属性列表 |

---

## 9. 对象复制 & Pickle
| 方法 | 调用时机 | 示例 |
|------|----------|------|
| `__copy__(self)` | `copy.copy(obj)` | 浅拷贝 |
| `__deepcopy__(self, memo)` | `copy.deepcopy(obj)` | 深拷贝 |
| `__getstate__` / `__setstate__` | Pickle 序列化/反序列化 | `pickle.dump/load` |
