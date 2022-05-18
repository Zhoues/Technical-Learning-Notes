# 常用API

[TOC]



## Scanner



- 从键盘输入单个数据:

```java
Scanner sc = new Scanner(System.in);
int i = sc.nextInt();
```

- 从文件输入多组数据：

```java
Scanner sc = new Scanner(new File("myNumbers"));
while (sc.hasNextLong()) {
    long aLong = sc.nextLong();
}
```

### 方法摘要

| 变量和类型 | 方法              | 描述                                                         |
| :--------: | ----------------- | :----------------------------------------------------------- |
| `boolean`  | `hasNextDouble()` | 如果使用 [`nextDouble()`](#nextDouble())方法将此扫描仪输入中的下一个标记解释为double值，则返回true。 |
| `boolean`  | `hasNextInt()`    | 如果使用 [`nextInt()`](#nextInt())方法将此扫描器输入中的下一个标记解释为默认基数中的int值，则返回true。 |
| `boolean`  | `hasNextLine()`   | 如果此扫描器的输入中有另一行，则返回true。                   |
|  `double`  | `nextDouble()`    | 将输入的下一个标记扫描为 `double` 。                         |
|   `int`    | `nextInt()`       | 将输入的下一个标记扫描为 `int` 。                            |
|  `String`  | `nextLine()`      | 使此扫描器前进超过当前行并返回跳过的输入。                   |


## String

- 字符串不变：字符串的值在创建之后是不能更改的

### 构造方法

| 方法名                      | 说明                                                  |
| --------------------------- | ----------------------------------------------------- |
| `public String()`           | 创建一个空白字符串对象，不含任何内容                  |
| `public String(char[] chs)` | 根据字符数组的内容，来创建字符串对象                  |
| `public String byte[] bys`  | 根据**字节数组的内容(ascii对应字符)**来创建字符串对象 |
| `String s = "abc"`          | 直接复制的方式创建字符串对象，内容就是`abc`           |

### 成员方法

| 返回值类型 | 方法名                                    | 描述                                                         |
| ---------- | ----------------------------------------- | ------------------------------------------------------------ |
| `char`     | `charAt(int index)`                       | 返回指定索引处的`char`值                                     |
| `int`      | `compareTo(String str)`                   | 按字典序比较连个字符串                                       |
| `int`      | `compareToIgnoreCase(String str)`         | 按字典序比较连个字符串,忽略大小写差异                        |
| `boolen`   | `equals(Object anObject)`                 | 将此字符串与指定对象进行比较                                 |
| `int`      | `indexOf(String str,[int fromIndex])`     | [从自定位置开始搜索]，返回指定字符第一次出现的此字符串中的索引 |
| `int`      | `lastIndexOf(String str,[int fromIndex])` | [从自定位置开始搜索]，返回指定字符最后一次出现的此字符串中的索引 |
| `int`      | `length`                                  | 返回字符串的长度                                             |
| `String[]` | `split(String regex)`                     | 将此字符串拆分为给定`regular rxpression`的匹配项             |
| `String`   | `substring(int beginIndex, int endIndex)` | 返回一个字符串，该字符串是此字符串的子字符串                 |

| 返回值类型      | 方法名        | 描述                                             |
| --------------- | ------------- | ------------------------------------------------ |
| `static String` | `toLowerCase` | 使用默认语言环境的规则将`String`所有字符转为小写 |
| `static String` | `toUpperCase` | 使用默认语言环境的规则将`String`所有字符转为大写 |



## StringBuilder

- **可变**的字符串类，我们可以把它看成一个容器，可以**进行链式操作**

### 构造方法

| 方法名                                 | 说明                                         |
| -------------------------------------- | -------------------------------------------- |
| `public StringBuilder()`               | 创建一个空白可变的字符串对象，不含有任何内容 |
| **`public StringBuilder(String str)`** | **根据字符串的内容，来创建可变字符串对象**   |

### 成员方法

| 返回值类型      | 方法名                                | 描述                                                         |
| --------------- | ------------------------------------- | ------------------------------------------------------------ |
| `StringBuilder` | `append`                              | 将参数的字符串表示形式追加到序列之中                         |
| `char`          | `charAt`                              | 返回指定索引处此序列的`char`值                               |
| `StringBuilder` | `delete(int start, int end)`          | 删除此序列的子字符串中的字符                                 |
| `StringBuilder` | `deleteCharAt(int index)`             | 阐述指定位置的`char`                                         |
| `int`           | `indexOf(String str,[int fromIndex])` | [从自定位置开始搜索]，返回指定字符第一次出现的此字符串中的索引 |
| `StringBuilder` | `insert(int offset,char c)`           | 将参数的表示形式插入到此序列中                               |
| `StringBuilder` | `reverse`                             | 反转字符串                                                   |
| **`String`**    | **`toString`**                        | **把`StringBuilder`转换为`String`**                          |



## ArrayList(List子类)

- `ArrayList<E>`是可调大小的数组实现，**\<E>是一种特殊的数据类型，泛型**

### 构造方法

| 方法名               | 说明                 |
| -------------------- | -------------------- |
| `public ArrayList()` | 创建一个空的集合对象 |

### 成员方法

| 返回值类型 | 方法名                       | 描述                                   |
| ---------- | ---------------------------- | -------------------------------------- |
| `void`     | `add([int index],E element)` | 将指定元素出啊如此列表[中的指定位置]   |
| `void`     | `clear`                      | 清空此列表                             |
| `booolean` | `equals(Object o)`           | 将指定对象与此列表进行比较以获得相等性 |
| `E`        | `get(int index)`             | 返回此列表中指定位置的元素             |
| `E`        | set(int index ,E element)    | 修改指定索引处的元素,返回被修改的元素  |
| `E`        | `remove(int index)`          | 删除此列表中指定位置的元素             |
| `boolean`  | `remove(Object o)`           | 删除指定的元素，返回是否删除成功       |
| `int`      | `size`                       | 返回列表中元素个数                     |



## Math

- 没有构造方法，如何使用类中的成员呢？	**看类的成员是否都是静态的，如果是，通过类名就可以直接调用**

- 这里对于API不再进行摘要，需要直接查API文档



## System

- 没有构造方法，直接用类名访问成员方法

### 成员方法

| 方法名              | 摘要                       |
| ------------------- | -------------------------- |
| `exit`              | 终止当前运行的`java`虚拟机 |
| `currentTimeMillis` | 以毫秒为单位返回当前时间   |

## Object

- 所有类的超类（父类）

### 成员方法

| 返回值类型 | 方法名     | 描述                                                        |
| ---------- | ---------- | ----------------------------------------------------------- |
| `String`   | `toString` | 返回对象的字符串表示形式。建议所有子列重写该方法,自动生成   |
| `boolean`  | `equals`   | 比较对象的hi否相等。默认比较地址,重写可以比较内容，自动生成 |



## 基本数据类型包装类（Integer）

### int转String

- 方法一

```java
int n = 100;
String s = "" + n;
System.out.println(s);
```

- 方法二

```java
int n = 100;
String s = String.valueOf(n)
System.out.println(s);
```

### String转int

- 方法一
- String ---- Integer ---- int

```java
String s = "100";
Integer i = Integer.valueOf(s);
int x = i.intValue();
System.out.println(x);
```

- 方法二
- String ---- int

```java
String s = "100";
int y = Ingeter.parseInt(s);
System.out.println(y);
```





### String,Integer,int转化关系表

| 类型转化           | 方法实现                                          |
| ------------------ | ------------------------------------------------- |
| String --- Integer | Integer i = Integer.valueOf(s);                   |
| Integer --- String | String s = i.toString()                           |
| int --- Integer    | Integer i = Integer.valueOf(n);(**可以自动装箱**) |
| Integer --- int    | int n = i.intValue();**（可以自动拆箱）**         |
| String --- int     | int n = Integer.parseInt(s);                      |
| int --- String     | String s = String.valueOf(n);                     |



## 日期类

### Date类

#### 构造方法

| 方法              | 描述                                                         |
| ----------------- | ------------------------------------------------------------ |
| `Date()`          | 分配一个`Date`对象并进行初始化，以便它**表示分配的时间**(是自动分配的时间) |
| `Date(long date)` | 分配一个`Date`对象,并将其初始化为从标准基时间起的毫秒数(是手动分配的时间) |

#### 成员方法

| 返回值类型 | 方法                 | 说明                                       |
| ---------- | -------------------- | ------------------------------------------ |
| `long`     | `getTime()`          | 获取的是日期对象从标准基时间到现在的毫秒值 |
| `void`     | `setTime(long time)` | 设置时间                                   |



### SimpleDateFormat类

- 重点学习日期格式化和解析

- 常用的模式字母及对应关系如下：

| 常用模式字母 | 对应关系 |
| ------------ | -------- |
| y            | 年       |
| M            | 月       |
| d            | 日       |
| H            | 时       |
| m            | 分       |
| s            | 秒       |

#### 构造方法

| 构造方法                    | 说明                                                         |
| --------------------------- | ------------------------------------------------------------ |
| `SimpleDateFormat()`        | 构造一个 `SimpleDateFormat`使用默认模式和日期格式符号默认 `FORMAT`区域设置。 |
| `SimpleDateFormat(pattern)` | 构造一个 `SimpleDateFormat`使用自定义的日期格式符号          |

#### 成员方法

| 返回值类型 | 方法                 | 说明                          |
| ---------- | -------------------- | ----------------------------- |
| String     | format(Date date)    | 将日期格式化成日期/时间字符串 |
| Date       | parse(String sourse) | 从给定字符串解析文本身成日期  |



### Calendar类

#### 创建对象

```java
Calendar c = Calendar.getInstance();	//多态的形式
```

#### 获取具体日历内容

```java
int year = c.get(Calendar.YEAR);
int month = c.get(Calendar.MONTH) + 1;
int date = c.get(Calendar.DATE);
```

#### 成员方法

| 返回值类型      | 方法                                 | 描述                                                     |
| --------------- | ------------------------------------ | -------------------------------------------------------- |
| `int`           | `get(int field)`                     | 返回给定日历字段的值                                     |
| `abstract void` | `add(int field, int amount)`         | 根据日历的规则，将指定的时间量添加或减去给指定的日历字段 |
| `final void`    | `set(int year, int month, int date)` | 设置当前日历的年月日                                     |



## 异常类

有两种处理方案：

- `try...catch...`
- `throws`



**try...catch...**

- 格式：

```java
try{
	//可能出现异常的代码
}catch (异常类名 变量名){
    // 异常处理的代码
}

//范例
try {
      int[] arr = {1, 2, 3};
      System.out.println(arr[3]);
} catch (ArrayIndexOutOfBoundsException e) {
      e.printStackTrace();
}
```



**throws**

- 格式：

```java
 throws 异常类名；
```

- **注意：这个格式更在方法的括号后面**



### Throwable

#### 成员方法

| 返回值类型 | 方法                | 说明                                            |
| ---------- | ------------------- | ----------------------------------------------- |
| `String`   | `getMessage()`      | 返回此`throwable`的详细消息字符串（异常的原因） |
| `String`   | `toString()`        | 返回此可抛出 的简短描述                         |
| `void`     | `printStackTrace()` | 把异常的错误消息输出在控制台                    |

### 自定义异常

格式：

```java
public class 异常类名 extends Exception{
	无参构造
	带参构造
}
```



## Collection(单例集合)

- 是单例集合的顶层接口，它表示一组对象，这些对象也称为`Collection`的元素

### 基本成员方法

| 返回值类型 | 方法                 | 说明                             |
| ---------- | -------------------- | -------------------------------- |
| `boolean`  | `add(E e)`           | 添加元素                         |
| `boolean`  | `remove(Object o)`   | 从集合重删除指定元素             |
| `void`     | `clear()`            | 清空集合中的元素                 |
| `boolean`  | `contains(Object o)` | 判断集合中是否存在指定的元素     |
| `boolean`  | `isEmpty()`          | 判断集合是否为空                 |
| `int`      | `size()`             | 集合的长度，就是集合中元素的个数 |

### 迭代器

`Iterator`：迭代器，集合的专用遍历方式

- `Iterator <E> iterator()`：返回此集合中元素的迭代器，通过集合的`iterator()`方法得到



`Iterator`中常用的方法:

- `E next()`：返回迭代中的下一个元素
- `boolean hasNext()`：如果迭代具有更多元素，则返回true

```java
Collection<String> c = new ArrayList<String>();
Iterator<String> it = c.iterator();
while(it.hasNext())
{
	System.out.println(it.next());
}
```



## List(Collection子类)

- **允许索引访问或修改**
- **允许重复的元素**

### 特有成员方法

| 返回值类型 | 方法                         | 描述                                  |
| ---------- | ---------------------------- | ------------------------------------- |
| `void`     | `add([int index],E element)` | 将指定元素出啊如此列表[中的指定位置]  |
| `E`        | `get(int index)`             | 返回此列表中指定位置的元素            |
| `E`        | set(int index ,E element)    | 修改指定索引处的元素,返回被修改的元素 |
| `E`        | `remove(int index)`          | 删除此列表中指定位置的元素            |

### 迭代器 ListIterator

`ListIterator`：列表迭代器

- 通过`List`集合的`listIterator()`方法得到，所以说它是`List`集合特有的迭代器
- **允许沿任意方向遍历列表的迭代器**

`ListIterator`中常用的方法:

- `E next()`：返回迭代中的下一个元素
- `boolean hasNext()`：如果迭代具有更多元素，则返回true

- `E pervious()`：返回迭代中的上一个元素
- `boolean hasPervious()`：如果迭代具有更多元素，则返回true
- `void add/remove/set(E e)`：将指定元素插入/删除/修改列表

## LinkedList(List子类)



### 特有成员方法

| 返回值类型 | 方法名          | 说明                             |
| ---------- | --------------- | -------------------------------- |
| `void`     | `addFirst(E e)` | 在列表开头插入指定元素           |
| `void`     | `addLast(E e)`  | 将指定的元素追加到此列表的末尾   |
| `E`        | `getFirst()`    | 返回此列表的第一个元素           |
| `E`        | `getLast()`     | 返回此列表的最后一个元素         |
| `E`        | `removeFirst()` | 从此列表中删除并返回第一个元素   |
| `E`        | `removeLast()`  | 从此列表中删除并返回最后一个元素 |



## Set(Collection子类)

- **不允许索引访问或修改**
- **不允许重复元素**



## HashSet(Set子类)

- 要保证元素的唯一性，需要重写`hashCode()`和`equals()`
- 内部数据结构就是**哈希表**



## LinkedHashSet(HashSet子类)

- 哈希表和链表实现的`Set`接口，具有可预测的迭代次序
- 链表表示元素有序，也就是说元素的存储和取出顺序是一致的
- 哈希表保证元素唯一，也就是说没有重复元素



## TreeSet(Set子类)

- 元素有序，这里的顺序不是指存储和取出顺序，而是按照一定的规则进行排序，具体排序方式取决于构造方法
- `TreeSet()`：根据其元素的自然顺序进行排序
- `TreeSet(Comparator  omparator)`：根据指定的比较器进行排序

### 自然排序Comparable的使用

- 重写接口

```java
public class Student implements Comparable<Student>{
    private String studentID;
    private String age;

    @Override
    public int compareTo(Student o) {
        if(this.addr!=o.age) {
            return this.age.compareTo(o.age) ;
        }else{
            return this.studentID.compareTo(o.studentID);
        }
    }
}
```



## Map(双例集合)



### 基本成员方法

| 返回值类型 | 方法                        | 说明                             |
| ---------- | --------------------------- | -------------------------------- |
| `V`        | `put(K key, V value)`       | 添加元素                         |
| `v`        | `remove(Object key)`        | 根据键删除键值对元素             |
| `void`     | `clear()`                   | 移除所有键值对元素               |
| `boolean`  | `containsKet(Object key)`   | 判断集合是否包含指定的键         |
| `boolean`  | `containsValue(Object key)` | 判断集合是否包含指定的值         |
| `boolean`  | `isEmpty()`                 | 判断集合是否为空                 |
| `int`      | `size()`                    | 集合的长度，即集合中的键值对个数 |

| 返回值类型                 | 方法              | 说明                         |
| -------------------------- | ----------------- | ---------------------------- |
| `V`                        | `get(Object key)` | 更具键获取值                 |
| `Set<K>`                   | `keySet()`        | 获取所有键的集合             |
| `Collection<V>`            | `values()`        | 获取所有值的集合             |
| **`Set<Map.Entry<K,V> >`** | **`enterSet()`**  | **获取所有键值对对象的集合** |
| `K____Map.Entry<K,V> `     | `getKey()`        | 返回与此条目对应的键。       |
| `V____Map.Entry<K,V> `     | `getValue()`      | 返回与此条目对应的值。       |

