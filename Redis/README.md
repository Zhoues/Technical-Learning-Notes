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



**应用：验证码**

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

```
set key value
expire key 5
pexpire key 5
```

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

**应用场景：生产者消费者模型——邮件发送队列**

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

## 位图

**应用：统计用户上线天数**

- SETBIT key offset value
  - 修改某个位置上的二进制值
  - 参数
    - offset	-	偏移量	从0开始
    - value     -    0或者1
- GET key offset
  - 获取某一位上的值

- BITCOUNT  sey start end
  - 统计键所对应的值中有多少个1
  - 参数
    - start/end 代表的是字节索引

## 哈希

就是一个小字典

1. 由field和关联的value组成的键值对
2. field和value是字符串类型
3. 一个hash中最多包含2^32-1个键值对

优点：

1. 节约内存空间-特定条件下【1. 字段小于512 	2. value不能超过64字节】
2. 可按需获取字段的值

缺点（不适合hash情况）

1. 使用过期键功能：键过期功能只能对键进行过期操作，而不能对散列的字段进过期操作
2. 存储消耗大于字符串结构



**应用：缓存个人信息**

### 常用命令

- HSET key field value
- HSETNX key field value
  - 设置单个字段
- HMSET key field value field value
  - 设置多个字段
- HLEN key
  - 返回字段个数



- HGETALL key
  - 返回该键值对应的结构



- HKEYS key
  - 返回所有字段名
- HVALS key
  - 返回所有值
- HDEL key field
  - 删除指定字段
- HINCRBY key field increment
- HINCRBYFLOAT key field increment
  - 在字段对应值上进行整数增量运算



## 集合

1. 无序、去重
2. 元素是字符串类型
3. 最多包含2^32-1个元素

**应用：**

1. **社交类平台：共同好友——交集**
2. **纯随机类抽奖**
3. **防止元素重复**
4. **黑/白名单**

### 常用命令

- SADD key member1 member2
  - 添加一个或者多个元素，自动去重
  - 返回值为成功插入到集合的元素个数
- SEMEBERS key
  - 查找集合中所有的元素

- SREM key member1 member2
  - 删除一个或者多个元素，元素不存在自动忽略



- SISMEMBER key member
  - 元素是否存在
- SRANDMEMBER key [count]
  - 随机返回集合中指定个数的元素，默认为1个
- SPOP key [count]
  - 弹出成员
- SCARD key
  - 返回集合中元素的个数



- SMOVE source destination member
  - 把元素从源集合移动到目标集合
- SDIFF key1 key2
  - 差集（number1   123     number2    124 结果为3）
- SDIFFSTORE destination key 1 key 2
  - 差集保存到另一个集合中



- SINTER key1 key2
- SINTERSTORE destination key 1 key 2
  - 交集

- SUNION key1 key2
- SUNION STORE destination key 1 key 2
  - 并集



## 有序集合

1. 有序、去重
2. 元素是字符串类型
3. 每个元素都关联着一个浮点数分值（score）,并按照分值从小到达的顺序排序集合中的元素（分值可以相同）
4. 最多包含2^32-1个元素



### 常用命令

- ZADD key score member
  - 在有序集合中添加一个成员 
  - 返回值：成功插入到集合中的元素个数

- ZRANGE key start stop [withscores]
  - 查看指定区间元素（升序）
- ZREVRANGE key start stop [withscores]
  - 查看指定区间元素（降序）

- ZSCORE key member
  - 查看指定元素的分值

- ZRANK key member
  - 查看排名（从0开始）（升序）

- ZREVRANK key member
  - 查看排名（从0开始）（降序）



- ZRANGEBYSCORE key min max [withscores] [limit offset count]
  - 参数说明：
    - min/max：最小值/最大值区间，默认闭区间（大于等于或者小于等于）
      - (min：可开启开区间即（大于或小于）
    - offset：跳过多少元素
    - count：返回几个
    - limit：选项和mysql一样



- ZREM key member
  - 删除成员
- ZINCRBY key increment member
  - 增加或者减少分值
- ZREMRANGEBYSCORE key min max
  - 删除指定区间内的元素（默认闭区间，可做开区间）
- ZCARD key
  - 返回集合元素中的个数
- ZCOUNT key min max
  - 返回指定范围中元素的个数（默认闭区间，可做开区间）



- ZUNIONSTORE destination numkeys key1 key2 [weights weight] AGGREGATE [SUM|MIN|MAX]

  - 并集
  - 参数
    - numkeys：参与合并的有序集合个数
    - weights：参与合并的有序集合的权重
    - AGGREGATE：聚合方式

- ZINTERSTORE destination numkeys key1 key2 [weights weight] AGGREGATE [SUM|MIN|MAX]

  - 交集

  



# 事务

redis 是弱事务型的数据库，并不具备ACID的全部特性

redis具备隔离性：事务中的所有命令会被序列化，按顺序执行，在执行过程中不会被其他客户端发送来的命令打断

不保证原子性：redis中的一个事务中如果存在命令执行失败，那么其他命令依然会被执行，没有回滚机制

## 基本命令

1. MULTI	开启事务
2. 命令1……
3. 命令2……
4. EXEC     提交到数据库执行
5. DISCARD    取消事务



# 数据持久化

## RDB

1. 保存真实的数据
2. 将服务器包含的所有数据库数据以二进制文件的形式保存到硬盘里面
3. 默认文件名：`/var/lib/redis/dump.rdb`
4. 文件名及目录可在配置文件中修改【`/etc/redis/redis.conf`】
   1. 263行：`dir /var/lib/redis` 表示rdb文件存放路径
   2. 253行：`dbfilename dump.rdb` 文件名

### 触发方式

#### 方式一：redis终端中使用SAVE或者BGSAVE命令

特点：

1. 执行SAVE命令过程中，redis服务器将被堵塞，无法处理客户端发送的命令请求，在SAVE命令执行完毕之后，服务器才会重新开始处理客户端发送的命令请求
2. 如果EDB文件已经存在，那么服务器将会自动使用新的RDB文件替代旧的RDB文件



BGSAVE执行流程如下：

1. 客户端发送 BGSAVE 给服务器
2. 服务器马上返回 Background saving started 给客户端
3. 服务器 fork() 子进程做这件事
4. 服务器继续提供服务
5. 子进程创建完RDB文件后再告知Redis服务器



SAVE 比 BGSAVE快，因为需要创建子进程，消耗额外的内存

说明：可以通过查看日志文件来查看redis的持久化过程

logfile位置：`/var/log/redis/redis-server.log`



#### 方式二：自动触发SAVE或者BGSAVE

218行：`save 300 10`

表示如果距离上一次创建EDB文件已经过去了300s，并且服务器的所有数据库总共已经发生了不少于10次的修改，你们自动执行BGSAVE命令

每次创建RDB文件之后，服务器为实现自动持久化而设置的时间计数器和次数计数器就会被清零，并重新考试计数，所以多个保存条件的效果不会叠加



#### 方式三：自动关闭redis



## AOF

1. 存储的是命令，而不是真实的数据
2. 默认不开启



**开启方式：修改配置文件**

1. `/etc/redis/redis.conf`
   1. 672 行：appendonly yes     把no改成yes
   2. 676行： appendfilename "appendonly.aof"
2. 重启服务 `sudo /etc/init.d/redis-server restart`



**执行原理**：

1. **每当有修改数据库的命令被执行时发生**
2. 因为AOF文件里面存储了服务器执行过的所有数据库修改的命令，所以给定一个AOF 文件，服务器只需要重新执行一遍AOF文件里面包含的所有命令，就可以达到还原数据库的目的
3. 用户可以根据自己的需要对AOF持久化进行调整，让redis再遭遇意外停机的时候不会丢失任何数据，或者只丢失一秒钟的数据，这比RDB持久化丢失的数据要少得多

**特殊说明：**

1. AOF 持久化：当一条命令真正的被写入到磁盘里面时，这条命令才不会应位停机二意外丢失
2. AOF 持久化再遭遇停机时丢失命令的数量，取决于命令被写入到硬盘的时间
3. 越早将命令写入到磁盘，发生意外停机时丢失的数据就越少，反之亦然

**干预操作系统配置：**打开配置文件：`/etc/redis/redis.conf`

1. 701行：alwarys

   服务器每写入一条指令，就会将缓冲区里面的命令写入到硬盘里面，服务器就算意外停机，也不会丢失任何已经成功执行的命令数据

2. 702行：everysec   （默认）

​		服务器每一秒将缓冲区里面的命令写入到磁盘里面，这种模式下，服务器及时遭遇意外停机，最多只丢失1s的数据

3. 703行：no

​		服务器不主动将命令写入磁盘，有操作系统决定何时将缓冲区里面的命令写入到磁盘里面，丢失命令数量不确定



**AOF重写**

由于AOF的文件中有很多冗余命令，系统会自动合并重写



### 触发AOF重写方式

1. 客户端向服务器发送BGREWRITEAOF命令

2. 修改配置文件让服务器自动执行BGREWRITEAOF命令

   1. auto-aof-rewrite-percentage 100

   2. auto-aof-rewrite-min-size 64mb

      解释：只有当AOF文件的增量大于100%时才会进行重写，也就是大一倍的时候才会触发



# 主从复制

高可用——是系统框架设计中必须考虑的因素之一，它通常是指，通过设计减少系统不能提供服务的时间

目标：消除基础架构中的单点故障

- redis单进程单线程模式，如果redis进程挂掉，相关依赖的服务就难以正常服务
- redis高可用方案——主从搭建 + 哨兵



设计模式：

1. 一个redis服务可以又多个该服务的复制品，这个redis服务称为master，其他复制品成为slaves
2. master会一直将自己的数据更新同步给slaves，保持主从同步
3. 只有master可以执行写命令，slave只能执行读命令

作用：分担了读的压力（高并发）；提高可用性

原理：从服务器执行客户端发送的读命令，客户端可以连接slaves执行读请求，来降低master的读压力



## Linux命令实现

命令：`redis-server --slaveof <master-ip> <master-port> --masterauth <master password>`

## Redis命令实现

1. slaveof IP PORT 成为谁的从
2. slaveof no one 自封为王



# 哨兵

## 基本概念

1. Sentinel会不断检查Master和Slaves是否正常
2. 每一个Sentinel可以监控任意多个Master和该Master下的Slaves



原理：正如其名，哨兵进程定期与redis主从进行通信，当哨兵认为redis主“阵亡”后【通信无返回】，自动将切换工作完成

## 安装和使用哨兵

```shell
# 1. 安装redis-sentinel
sudo apt install redis-sentinel
验证： sudo /etc/init.d/redis-sentinel stop

# 2. 新建配置文件setinel.conf
port 26379
sentinel monitor tedu 127.0.0.1 6379 1

# 3. 启动sentinel
方式一： redis-sentinel sentinel.conf
方式二： redis-server sentinel.conf --sentinel

# 4. 将master的redis服务终止，查看是否会提上为主
sudo /etc/init.d/redis-server stop

# 发现提升6381为master，其他两个为从
```

## 配置文件解读

```shell
# sentinel监听端口，默认是26379,可以修改
port 26379
# 告诉sentinel去监听地址为ip:port的一个master,这里的master-name可以自定义，quorum是一个数字，指明当前有多少个sentinel认为一个master失效时，master才真正失败
sentinel monitor <master-name> <ip> <redis-port> <quorum>

# 如果master有密码，则需要额外添加该配置
sentinel auth-pass <master-name> <password>

# master多久失联才认为时不可用了，默认时30s
sentinel down-after-milliseconds <master-name> <milliseconds>
```







## 创建配置文件

```shell
# 每个redis服务，都有1个和他对应的配置文件
# 两个redis服务
1. 6379 -> /etc/redis/redis.conf
2. 6300 -> /home/ubuntu/redis_6300.conf

# 修改配置文件
vi redis_6300.conf
slaveof 127.0.0.1 6379
port 6300
# 启动redis服务
redis-server redis_6300.conf
# 客户端连接测试
redis-cli -p 6300

```





# 使用pyredis操作redis

## 操作数据库

**和命令行操作基本一致**

```python
import redis
# 创建数据库连接对象
r = redis.Redis(host='r-2zejdc10y47fr3t4kipd.redis.rds.aliyuncs.com',
                port=6379, db=0, password='buaa(2021)')
```



## 操作事务

### pipeline流水线执行批量操作

**批量执行redis命令，减少通信io**

原理：效仿redis的事务，客户端将多个命令打包，一次性通信发送给redis，可明显降低redis服务的请求数

注意：

1. 此为客户端技术
2. 如果一组命令中，一个命令需要上个命令的执行结果才可以执行，则无法使用该技术



```python
# 创建连接池并连接到 redis
pool = redis.ConnectionPool(host='',port=6379, db=0, password='')
r = redis.Redis(connection_pool=pool)

pipe = r.pipeline()
....
pipe.execute()

```



### pipeline流水线执行事务

python操作事务，需要依赖流水线技术

```python
with r.pipeline(transaction=True) as pipe:
    pipe.multi()
	...
    values = pipe.execute()

```



### watch 乐观锁

事务过程中，可以对指定key进行监听，命令提交时，若被监听key对应的值未被修改，事务方可提交成功，否则失败

```python
with r.pipeline(transaction=True) as pipe:
    while True:
        try:
            key = ....
            pipe.watch(key)
            value = int(r.get(key))
            value *= 2

            pipe.multi()
            r.set(key,value)
            pipe.execute()
            break
        except redis.WatchError:
            continue
	return int(r.get(key))
```



