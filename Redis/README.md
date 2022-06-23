# Redis

[TOC]



# 应用场景

1. 缓存
2. 并发计数：由于redis本身是单进程单线程的，可以有效解决并发请求技术场景；例如微博点赞
3. 排行榜：各大实时排行榜-如电商/游戏中的排行
4. 生产者消费者模型

# 安装&连接

**Ubuntu**

安装：sudo apt-get install redis-server

​		安装后，Ubuntu会将redis列为开机自动启动项

服务端启动/停止/重启：

​		`sudo /etc/init.d/redis-server status |  start | stop | restart`

客户端连接

​		`redis-cli -h IP地址 -p 6379 -a 密码`



## 配置文件中添加密码

requirepass 密码

重启服务：`sudo /etc/init.d/redis-server status restart`

重新链接：`redis-cli -h IP地址 -p 6379 -a 密码`



## 重启redis失败

大概率为配置文件配置有误导致

​		解决方案一：终端中直接执行` redis-server /etc/redis/redis.conf` 启动服务，若有报错，按照报错提示修改`redis.conf`

​		解决方案二：还原备份后的配置文件



## 远程链接

注释掉本地IP地址绑定

​	69行：`#bind 127.9.9.1 ::1`

关闭保护模式（把`yes`改为`no`）

​	88行：`protected-mode no`

重启服务

​	`sudo /etc/init.d/redis-server status restart`



# Redis基础命令（无关数据类型）

即无关于具体的数据类型

- select number

说明：切换数据库（默认redis有16个数据库，0-15为具体数据库的编号，默认进入redis的编号为0）

- info

说明：查看redis服务的整体情况

- keys表达式

说明：查找所有符合给定模式的key

样例：

​	KEYS * ：匹配数据库中所有的key

​	KEYS h?llo ：匹配hello,hallo和hxllo等。

​	KETS h*llo ：匹配hllo和heeeeello等

特殊说明：正式环境中，请勿使用此命令；由于redis单进程单线程，当key很多时，当前命令可能堵塞redis

- type key
  - 返回当前键的数据类型
- exists key
  - 返回当前键是否存在
  - 返回值：1表示当前key存在；0表示当前key不存在
- del key
  - 删除key
- rename key newkey
  - 重命名当前key的名字
- flushdb
  - 清除当前所在数据库数据
- flushall
  - 清楚所有数据库的数据



# 数据类型

## 字符串

字符串、数字，都会转为字符串来存储

以二进制的方式存储在内存中



**注意：**

key命名规范

​	可采用	- wang:email

key命名原则

1. key值不宜过长，消耗内存，且在数据中查询这里简直的计算成本高
2. 不宜过短，可读性差
3. 一个字符串类型的值最多能存储512M内容



### 常用命令

- set key value nx ex
  - 设置一个字符串的key
  - 特殊参数
    - nx => not exist 代表当key不存在时，才会存储这个key
    - ex => expire 过期时间，单位s
- get key
  - 获取key的值
  - 返回值：key的值或者'nil'
- strlen key
  - 获取key存储值的长度
- getrange key start stop
  - 获取字符串指定范围切片内容【包括start stop】
- setrange key index value
  - 字符串从index索引值开始，用value替换原内容中value长度大小字符串；返回最新长度

- mset key1 value1  key2 value2  key3 value3
  - 批量添加key和value
- mget key1 key2 key3
  - 批量获取key 的值

- incrby key 步长
  - 将key增加指定步长
- decrby key 步长
  - 将key减少指定步长
- incr key
  - +1操作
- decr key
  - -1操作
- incrbyfloat key step



### 再谈过期时间

默认情况下,key没有过期时间，需要手动指定

方案一：直接使用set的ex参数

​	set key value ex 3

方案2：使用expire通用命令

	set key value
	expire key 5
	pexpire key 5

### 检查过期时间

- ttl  key 

  - 查看过期时间		

  - 返回值：

    -  -1 ：代表当前key没有过期时间

    - \> 0 ：代表当前key的剩余存活时间

    -  -2 ： 当前key不存在

- persist key
  - 删除过期时间
  - 把带有过期时间的key变成永久不过期
  - 返回值：
    - 1：代表删除过期时间成功
    - 0：代表当前key没过期时间或者不存在



### 删除机制

每个redis数据库中，都会有一个特殊的容器负责存储带有过期时间的key以及它对应的过期时间，这个容器称之为”过期字典“

针对过期字典中的key，redis结合惰性删除和定期删除两大机制，有效删除过期数据



## 列表

1. 元素是字符串类型
2. 列表头尾增删快，中间增删慢，增删元素是常态
3. 元素可重复
4. 最多可包含2^32-1个元素
5. 索引同python列表



### 常用命令

- LPUSH key value1 value2
  - 从列表头部插入压入元素
  - 返回list最新的长度
- RPUSH key value1 value2
  - 从列表尾部插入压入元素
  - 返回list最新的长度

- LRANGE key start stop
  - 查看列表中的元素

- LLEN key
  - 获取列表长度

- RPOPLPUSH src dst
  - 从列表src尾部弹出一个元素，压入到列表dst的头部
  - 返回被弹出的元素
- LINSERT key after|before value newvalue
  - 在列表指定元素后|前插入元素
  - 返回
    - 如果命令执行成功，返回列表的长度
    - 如果没有找到pivot，返回-1
    - 如果key不存在或者为空列表，返回0



- LPOP
  - 从列表头部弹出1个元素
- RPOP
  - 从列表尾部弹出1个元素
- BLPOP key timeout
  - 列表头部，阻塞弹出，列表尾空时堵塞
- BRPOP key timeout
  - 列表尾部，阻塞弹出，列表尾空时堵塞



- LREM key count value
  - 删除指定元素
  - count > 0：表示从头部开始向表尾搜索，移除与value相等的元素，数量为count
  - count < 0：表示从尾部开始向表头搜索，移除与value相等的元素，数量为count
  - count = 0：以处表中所有与value相等的值
  - 返回：被移除的元素数量

- LTRIM key start stop
  - 保留指定范围内的元素
  - 场景：保留评论最后500条

- LSET key index newvalue
  - 设置list指定索引的值



# 使用pyredis操作redis

## 操作流程

### 建立连接对象

```python
import redis
# 创建数据库连接对象
r = redis.Redis(host='r-2zejdc10y47fr3t4kipd.redis.rds.aliyuncs.com',
                port=6379, db=0, password='buaa(2021)')
```

