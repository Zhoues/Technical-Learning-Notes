# MySQL

[TOC]



## MySQL产品的介绍和安装



### MySQL服务的启动和安装

- 方式一：计算机——右击管理——服务
- 方式二：通过管理员身份运行
  - net start 服务名（启动服务）
  - net stop 服务名（停止服务）

### MySQL服务的登录和退出   

- 方式一：通过mysql自带的客户端
  - 只限于root用户

- 方式二：通过windows自带的客户端
  - 登录：mysql 【-h主机名 -P端口号 】-u用户名 -p密码
  - 退出：exit或ctrl+C

### **MySQL的常见命令** 

```mysql
1.查看当前所有的数据库
	show databases;

2.打开指定的库
	use 库名

3.查看当前库的所有表
	show tables;

4.查看其它库的所有表
	show tables from 库名;

5.查看当前所在库
	select database();

6.创建表
	create table 表名(

		列名 列类型,
		列名 列类型，
		。。。
	);

7.查看表结构
	desc 表名;

8.查看表中数据
	select * from 表名;

9.查看服务器的版本
	方式一：登录到mysql服务端
		select version();
	方式二：没有登录到mysql服务端
		mysql --version
		或
		mysql --V
		
```

### MySQL的语法规范

1. 不区分大小写,但建议关键字大写，表名、列名小写
2. **每条命令最好用分号结尾**
3. 每条命令根据需要，可以进行缩进 或换行
4. 注释
   	单行注释：#注释文字
   	单行注释：-- 注释文字
   	多行注释：/* 注释文字  */



## **SQL语言**

### SQL的语言分类

​	DQL（Data Query Language）：数据查询语言
​		select 
​	DML(Data Manipulate Language):数据操作语言
​		insert 、update、delete
​	DDL（Data Define Languge）：数据定义语言
​		create、drop、alter
​	TCL（Transaction Control Language）：事务控制语言
​		commit、rollback

## 查询 DQL

### 基础查询

```mysql
# 语法
select 查询字段 from 表名;	# 查询单个字段
select 查询字段,查询字段,查询字段... from 表名;	# 查询多个字段
select * from 表名;		  	# 查询所有字段
select 100;					# 查询常量
select 'john';				# 查询字符串（mysql不区分字符和字符串）
select 100*98;				# 查询表达式
select VERSION();			# 查询函数

# 注意
查询字段可以加上着重号(``)，区别于关键字

# 特点
1. 查询列表可以是：表中的字段，常量值，表达式，函数
2. 查询的结果是一个虚拟的表格

# 起别名(使用as 或者 使用空格)
select 查询列表 as output;	# as起别名
select 查询字段1 as name1, 查询字段2 as name2 from 表名;	# as起别名
select 查询字段1  name1, 查询字段2  name2 from 表名;	# 空格起别名
select 查询字段 as "out put" from 表名;	# 对于有特殊字符的别名用字符串

# 去重
select distinct 查询字段 from 表名;

# +号的作用
select 100+90;		# 两个操作数都为数值型，则做加法运算
select '123' + 90;	# 如果一方为字符型，试图讲字符型数值转换为数值型
					# 如果转换成功，则继续做加法;如果失败，则将字符型转为0
select null + 10;	# 只要其中一方为null，则结果肯定为null

# 字段（字符串）拼接--concat
select concat(str1,str2) from 表名;	# 拼接字符串
```



### 条件查询

```mysql
# 语法
select 查询字段 from 表名 where 筛选条件;

# 分类
1.按条件表达式筛选
	条件运算符: > < = != <> >= <=
2.按逻辑表达式筛选
	逻辑运算符: && || ! and or not	# 可以使用括号
3. 模糊查询
	模糊查询 like   between and  in  is null|is not null

like:	一般和通配符搭配使用
			% 任意多个字符，包含0个字符
			_ 任意单个字符，支持转义(可以通过escape转义)
				... where last_name like '_$_%' escape '$';
between and				包含临界值，不可以调换顺序
in						列表的值类型必须一致或者兼容，不支持通配符
is null|is not null		=或者<>不能判断null值，这两者可以

# 安全等于 <=>
	万能等号，可以判断null，缺点是不能下意识知道是等于
```



### 排序查询

```mysql
# 语法
select 查询字段 from 表名 where 筛选条件 order by 排序列表 【ASC/DESC】;

# 排序方法
1. 按字段排序
	select 查询字段 from 表名 order by 字段 【ASC/DESC】;
2. 按表达式排序
	select 查询字段 from 表名 order by 表达式 【ASC/DESC】;
3. 按别名排序
	select 查询字段 AS 别名 from 表名 order by 别名 【ASC/DESC】;
4. 按函数排序
5. 多级排序
```



### 分组查询(使用分组函数)

```mysql
# 使用 group by 			遇到"每个"的时候，group by就放"每个"之后的名词
# 语法
select 分组函数，列（要求出现在group by的后面）
from 表
[where 筛选条件]
group by 分组的列表
[order by 子句]

# 注意事项：
	查询列表必须特殊，要求是分组函数和group by后出现的字段

# 特点
	# 分组筛选分成两类——分组前筛选，分组后筛选
	分组前筛选 —— 原始表(WHERE) —— group by 子句的前面 —— WHERE
	分组后筛选 —— 分组后的结果集 —— group by 子句的后面  —— HAVING
	
	# 分组函数做条件肯定是放在HAVING子句中的 
	
# 复杂筛选 使用HAVING
eg:查询哪个部门的员工个数 > 2
select COUNT(*), DEPARTMENT_ID
FROM employees
GROUP BY department_id
HAVING COUNT(*) > 2;
```

### 连接查询

含义：又称多表查询，当查询字段来自于多个表时，就会用到连接查询

#### 内连接： 使用多个表中共同的列（sql92）

```mysql
1	等值连接

# 多表等值连接的结果为多表的交集部分
# n 表连接，至少需要n - 1 个连接条件
# 多表的顺序没有要求
# 一般需要给表取别名
# 可以搭配前面介绍的所有子句使用，比如排序，分组

1.1 等值连接样例
select last_name,department_name
from emloyees,departs
where employess.department_id = departments.department_id

1.2 可以给表起别名
select e.last_name,d.department_name
from emloyees e,departs d
where e.department_id = d.department_id

1.3 表的顺序可以调换
select e.last_name,d.department_name
from departs d, emloyees e
where e.department_id = d.department_id

1.4 可以加筛选 （where使用完毕之后可以使用and）
select last_name, department_name, commission_pct
from employees e , departments d
where e.department_id = d.department_id
and commission_pct is not null ;

1.5 可以加分组 （对于连接查询的group by要小心，如果不好确定除了分组函数之外的列的关系，最好都加上）
select MIN(salary), d.manager_id, department_name
from employees e,departments d
where e.department_id = d.department_id
and commission_pct is not null
group by department_name,d.manager_id;

1.6 可以加上排序
select job_title,count(*)
from jobs b, employees e
where b.job_id = e.job_id
group by job_title
order by count(*) desc ;


2	非等值查询
# where 中 不是等号连接

3	自连接
# 连接的表是本身 （这个时候就需要靠给表取别名来分辨）
select e.employee_id,e.last_name,d.employee_id,d.last_name
from employees e , employees d
where e.manager_id = d.employee_id
```

#### 内连接，外连接，交叉连接（sql99）

```mysql
语法：
	select 查询列表
	from	表1	别名	【连接类型】
	join	表2	别名
	on	连接条件
	【where		筛选条件】
	【group by	分组】
	【having		筛选条件】
	【order by	排序列表】
	
连接类型——分类：
内连接	——————	inner
外连接
	左外	——————	left outer
	右外	——————	right outer
	全外	——————	full outer
交叉连接   ——————  cross


1	内连接
select count(*) , department_name
from departments d
inner join employees e on d.department_id = e.department_id
group by department_name
having count(*) > 3
order by count(*) desc ;

2	外连接
# 查询一个表中有，另外一个表中没有的，使用外连接

3 	交叉连接
# 就是笛卡尔乘积
```

### 子查询

```mysql
出现在其他语句中的 select 语句，称为子查询或内查询
外部的查询语句，称为主查询或外查询

分类
按子查询出现的位置：
	select 			后面		————		标量子查询
	from			后面		————		表子查询
	where或者having  后面		————	   标量子查询，列子查询，行子查询
	exists			后面（相关子查询）———— 	表子查询

按结果集的行列数不同：
	标量子查询	————	结果集只有一行一列		> < >= <= <>
	列子查询	————	结果集一列多行			in, any/some, all
	行子查询	————	结果有一行多列
	表子查询	————	结果集一般为多行多列
特点：
	子查询放在小括号内
	子查询一般放在条件的右侧
	标量子查询，一般搭配着单行操作符使用	————	< > <= >= = <>
	列子查询，一般搭配着多行操作符使用	————	IN,ANY/SOME,ALL
	
	
	
	
一,	where或having后面 ———— 标量子查询，列子查询，行子查询
	
# 谁的工资比Abel高	————	标量子查询 --- > < >= <= <>
select *
from employees
where salary > (select salary
                from employees
                where last_name = 'Abel');
                
# 列子查询，一般搭配着多行操作符使用	————	IN,ANY/SOME,ALL
IN/NOT IN		等于列表里中的任意一个
ANY/SOME		和子查询返回的某一个值比较
ALL				和子查询返回的所有值比较

# 行子查询，要求很多，不常使用	---		（...） = ( select ... )
# 查询员工编号最小并且工资最高的员工信息
SELECT *	FROM employees
WHERE (employee_id,salary) = (
    	SELECT MIN(employee_id),MAX(salary)
    	FROM	employees
);


二，	select 后面		————		标量子查询
三，	from后面			————		表子查询	/*将子查询结果充当一张表，要求必须取别名*/
# 查询每个部门的平均工资的工资等级
SELECT ag_dep.* , g.grade_level
FROM (
	SELECT AVG(salary) ag, department_id
    FROM employees
    GROUP BY department_id
) ag_dep
INNER JOIN job_grades g
ON ag_dep.ag BETWEEN lowest_sal AND highest_sal;

四，	exists后面（相关子查询）	/* 语法： exists(完整的查询语句)	结果：1/0 */
```

### 分页查询

```mysql
/*
应用场景： 当要显示的数据一页显示不全，需要分页提交sql请求
*/
语法：
	select 查询列表
	from	表1	别名	【连接类型】
	join	表2	别名
	on	连接条件
	【where		筛选条件】
	【group by	分组】
	【having		筛选条件】
	【order by	排序列表】
	limit offset,size;
	
	offset	要表示条目的其实索引（其实索引从0开始）
	size	要显示的条目个数

特点：
	1. limit语句放在查询语句的最后
	2. 公式
	要显示的页数page，每页的条目数size
	
	select 查询列表
	from 表
	limit (page-1)*size , size;
	
# 查询前五条员工信息
SELECT * FROM employees LIMIT 0,5;
```

### 联合查询

```mysql
/*
union 联合 合并： 将多条查询语句的结果合并成一个结果
*/
# 语法
查询语句1
union
查询语句2
union
...

# 应用场景：
要查询的结果来自于多个表，且多个表没有直接的连接关系，但查询的信息一致时

# 特点
1. 要求多条查询语句的查询列数时一致的！
2. 要求多条查询语句的查询的每一列的类型和顺序最好一致
3. union关键字默认去重，如果使用union all 可以包含去重项
```



## 增删改 DML

### 插入语句 insert

```mysql
# 语法一
表名	列名	新值
insert into 表名(列名,...)	values(值1,...);

# 注意事项
1. 插入的值的类型要与列的类型一致或者 兼容
2. 不可以为NULL的列必须插入值，可以为NULL的列插入值有两种方式
	方式一： 直接在values中填NULL即可
	方式二： 直接在表名中不填入这个字段
3. 列的顺序可以调换，但是值要求一一对应
4. 列数和值的个数必须一致
5. 可以省略列名，默认所有列，而且列的顺序和表中列的顺序一致


# 语法二
insert into 表名
set 列名 = 值, 列名 = 值,...

# 两种语法PK
1. 方式一可以进行插入多行，方法二不支持
2. 方式一支持子查询，方法二不支持
```

### 修改语句 update

```mysql
/*
1. 修改单表记录
2. 修改多表的记录【补充】
*/

# 一.语法
update 表名
set 列=新值, 列=新值, ...
where 筛选条件;

# 二.语法
sql92
update 表1 别名, 表2 别名
set 列 = 值,...
where 连接条件
and 筛选条件;

sql99
update 表1 别名
set 列 = 值,...
inner|left|right join 表2 别名
on 连接条件
where 筛选条件
```

### 删除语句 delete

```mysql
/*
方式一：delete
	单表删除
	多表删除
方式二：truncate
*/

# 一.语法
单表删除
delete from 表名 where 筛选条件

多表删除
sql92
delete 表1的别名，表2的别名	[写了哪一个的别名就删哪一个]
from 表1 别名, 表2 别名
where 连接条件
and 筛选条件

sql99
delete 表1的别名，表2的别名	[写了哪一个的别名就删哪一个]
from 表1 别名
inner|left|right join 表2 别名 on 连接条件
where 筛选条件;

# 二.语法
truncate from 表名;

# 两种语法PK
1. delete 可以加 where 条件, truncate 不能加
2. truncate删除效率高一丢丢
3. 假如要删除的表有自增长列，delete之后从断点开始,truncate由1开始
4. delete删除有返回值，truncate没有返回值
5. truncate 删除不能回滚， delete删除可以回滚
```



## 数据库定义语言 DDL

```mysql
库的管理	————	创建，修改，删除
表的管理	————	创建，修改，删除

创建	create
修改	alter
删除	drop
```



### 库的创建	create

```mysql
# 语法
create database [if not exists]	库名;
```

### 库的修改	alter

```mysql
# 修改库的字符集
alter database 库名 character set 字符编码;
```

### 库的删除	drop

```mysql
# 语法
drop database [if exists]	库名;
```

### 表的创建	create

```mysql
# 语法
create table 表名(
	列名	列的类型[长度		约束]	,
    列名	列的类型[长度		约束]	,
    ...
    列名	列的类型[长度		约束]	
);
```

### 表的修改	alter

````mysql
/*
alter table 表名 add|drop|modify|change column 列名 [列类型 约束]
*/

# 分类
1. 修改列名
alter table 表名 change column 旧列名 新列名 新类型

2. 修改列的类型或约束
alter table 表名 modify column 列名	新类型 约束

3. 添加列
alter table 表名 add column 新列名 新类型

4. 删除列
alter table 表名 drop column 列名

5. 修改表名
alter table 旧表名 rename to 新表名

# 添加约束 （列级修改使用modify,表级修改使用add）
添加非空约束：
	alter table 表名 modify cloumn 列名	新类型 not null
添加默认约束：
	alter table 表名 modify cloumn 列名	新类型 default 默认值
添加主键：
	列级约束
		alter table 表名 modify cloumn 列名	新类型 primary key
	表级约束
		alter table 表名	add primary key(主键列)
添加唯一
	列级约束
		alter table 表名 modify cloumn 列名	新类型 unique
	表级约束
		alter table 表名	add unique(主键列)
添加主键
	alter table 表名 add foreign key(外键名)	references 主键表(主键名)
	
# 删除约束
删除非空约束：
	alter table 表名 modify cloumn 列名	新类型 null
删除默认约束：
	alter table 表名 modify cloumn 列名	新类型 
删除主键：
	列级约束
		alter table 表名 modify cloumn 列名	新类型 
	表级约束
		alter table 表名	drop primary key
删除唯一
	alter table 表名 drop index 列名
删除外键
	alter table 表名 drop foreign key 从表外键名
````

### 表的修改	drop

```mysql
drop table 表名
```

### 表的复制	

```mysql
1. 仅仅复制表的结构
create table 新表 like 原表;

2. 复制表的结构 + 数据
create table 新表 
select * from 原表;

3. 只复制部分数据
create table 新表
select 列1， 列2....
from 原表
where ... 

4. 仅仅复制某些字段
create table 新表
select 列1，列2 ...
from 原表
where 0;
```

## 常见数据类型

```mysql
/*
数值型：
	整数
	小数：
		定点数
		浮点数
字符型：
	较短的文本：char,varchar
	较长的文本：text,blob(较长的二进制数据)
日期型
*/
一. 整形
分类：
tinyint, smallint, mediumint, int/integer, bigint
特点：
1. 如果不设置无符号还是有符号，默认是有符号，如果想设置无符号，需要添加unsigned关键字
2. 如果插入的数据超出临界范围，则会默认设置为临界值
3. 如果不设置长度，会有默认的长度，长度代表了显示的最大宽度，如果不够会用0在左边填充，但是搭配 zerofill使用

二. 小数
分类：
浮点型
float(M,D) double(M,D)
定点型
DEC(M,D) decimal(M,D)

特点：
(M,D) 		M	—— 整数部分+小数部分		D —— 小数部分		如果超出范围，会插入临界值
如果是decimal，则 M 默认是10， D 默认是 0
如果是float 和 double， 则会根据插入的数值的精度来决定精度

三. 字符型
较短的文本：char,varchar
较长的文本：text,blob(较长的二进制数据)

四. 日期型
分类
date datetime timestamp time year

```

### 常见约束

```mysql
# 含义：一种限制，用于限制表中的数据，为了保证表中的数据的准确性和可靠性
分类：六大约束
	not null:	非空，用于保证该字段的值不为空，比如姓名学号等
	default:	默认，用于保证该字段有默认值，比如性别
	primary key	主键，用于保证该字段的值具有唯一性并且非空，比如学号员工编号
	unique		唯一，用于保证该字段具有唯一性，可以为空，比如座位号
	check		检查约束，比如年龄性别
	foreign key	外键，用于限制两个表的关系，用于保证该字段的值必须来自于主表的关联列的值
				在从表中加入外键约束，用于引用主表中某列的值
添加约束的时机：
	1. 创建表时
	2. 修改表时
约束的添加分类：
	列级约束
		外键约束没有效果
	表级约束
		处理非空和默认其他都支持
		
# 一. 列级约束
	列名	列类型  列级约束	(外键约束无效)
# 二. 表级约束
# 语法：在各个字段的最下面
[constraint 约束名]	约束类型(字段名...)	多个字段名可以设置为组合键，但是不推荐
[constraint 约束名] FOREIGN KEY (外键字段) REFERENCES 主键表 (主键) # 外键相连

# 三. 外键详解
	1. 要求从表设置外键关系
	2. 从表的外键列的类型和主表的关联列的类型要求一致兼容，名称无要求
	3. 主表的关联列必须是一个 key (一般时主键或者唯一键)
	4. 插入数据时，先插入主表，再插入从表；删除的时候先删除从表，再删除主表
	
# 四. 标识列
	auto_increment	:	自增
	set auto_increment_increment = x	x为自增的间隔数
	对于第一个插入的值可以确定起始自增位置
    
    1. 标识列搭配的必须是一个key
    2. 一个表至多有一个标识列
    3. 标识列的类型只能是数值型
```



## 视图

### 视图的创建 create

```mysql
# 含义：虚拟表，和普通表一样使用，通过表动态生成的数据
# 只保留sql逻辑
# 语法
create view 视图名 as
DQL;

# 好处
1. 重用 sql 语句
2. 简化负责的 sql 操作，不比知道它的查询细节
3. 保护数据，提高安全性
```

### 视图的修改

```mysql
# 语法
create or replace view 视图名 as
DQL;

alter view 视图名 as
DQL;
```

### 视图的删除 和 查看

```mysql
# 删除语法
drop view 视图名,视图名...;

# 查看语法
desc/show create view 视图名;
```

### 视图的更新

```mysql
视图定义中含有下列语法成分时，DML会受到限制
JOIN、GROUP BY 等子句
DISTINCT等修饰词
SELECT子句中含有聚合函数
WHERE的子查询引用了FROM句子中的表
```



## 事务 TCL

```mysql
# 概念——事务：一个或一组sql语句组成一个执行单元，这个执行单元要么全部执行，要么全部不执行
# 特点——ACID
	1. 原子性——事务是一个不可分割的工作单位，事务中的操作要么都发生，要么都不发生
	2. 一致性——事务必须使数据库从一个一致性状态换到另外一个一致性状态
	3. 隔离性——一个事务内部的操作及其使用的数据对并发的其他事务是隔离的，并发执行的各个事务之间不能相互干扰
	3. 持久性——一个事务一旦执行成功，其对数据库中的数据改变是永久的
```

### 事务的创建

```mysql
# 隐式事务：事务没有明显的开启和结束的标记——比如insert,update,delete语句

# 显示事务：事务有明显的开启和结束的标记
	# 前提：必须先设置自动提交功能为禁用		set atorcommit = 0;

# 步骤
步骤1：开启事务
set autocommit=0;
[start transaction;]

步骤2：编写事务中的sql语句(select insert ypdate delete)
语句1；语句2；语句3；...

步骤3：结束事务
commit;		提交事务
rollback;	回滚事务
savepoint 节点名;	设置保存点
rollback to 节点;
```

### 设置隔离级别

```mysql
select @@tx_isolation;
set [session] transaction isolation level read committed;
set global transaction isolation level read committed;

# 事务的隔离级别		脏读			不可重复读			幻读
read uncommitted;	OK				OK				 OK
read committed;		X				OK				 OK
repeatable read:	X				X				 OK
serializable		X				X				 X

mysql 中默认是第三个级别		repeatable read
```

## 存储过程

### 存储过程介绍

#### SQL指令执行过程

![1](C:\Users\这是恩申的哟\Desktop\Technical-Learning-Notes\MySQL\picture\1.png)

从SQL执行的流程中我们分析存在的问题：如上图黄字所示

#### 什么是存储过程

将能够完成特定功能的SQL指令进行封装，编译之后存储在数据库服务器上，并且为之取一个名字

当客户端需要执行这个功能的时，不用编写SQL指令，直接通过封装的SQL指令的名字完成调用即可

### 存储过程的创建  create procedure

```mysql
# 语法
create procedure <存储过程名> (IN 输入参数.... ,OUT 输出参数...)
begin
	-- SQL
end;

#e.g 
create procedure proc_test(IN a int ,IN b int ,OUT c int)
begin
	SET c = a+b;
end;
```

### 调用存储过程	call

```mysql
# 语法
先定义一个变量，调用存储过程，显示变量
-- 定义一个变量
set @m = 0;
-- 调用存储过程
call <存储过程名>(输入参数.... ， @m);
-- 显示变量
select @m from dual;
```

### 存储过程中变量的使用		declare

```mysql
# 存储过程中变量分为两种——局部变量 和 用户变量

# 局部变量	 ——		定义在存储过程中的变量，只能在存储过程中使用，必须定义在存储过程开始的位置
declare 临时变量名 临时变量数据类型 default 默认值;

# 用户变量	 ——		相当于全局变量，定义的用户变量可以通过 select @attrName from dual;进行查询
# 特点	——	用户变量会存储在mysql数据的数据字典中(dual)，用户变量定义使用set关键字直接定义，需要以@开头
set @n = 1;

无论是局部变量还是用户变量，都需要使用 set 关键字修改值
```

### 存储过程中需要使用SQL语句	into

```mysql
#e.g 
create procedure proc_test(OUT c int)
begin
	select count(*)	INTO c from students;
end;
```

### 存储过程的参数	IN \ OUT \ INOUT

```mysql
# MySQL 存储过程的参数一共有三种 ：IN \ OUT \ INOUT

# 输入参数	——	在调用过程中传递数据给存储过程的参数(在调用的过程必须为具有实际值的变量 或者 字面值)

# 输出参数	——	将存储过程中产生的数据返回给过程的调用者，一个存储过程可以有多个多个输出参数

# 输入输出参数	——	统一输入参数和输出参数
```

### 存储过程中流程控制	分支语句和循环语句

#### 分支语句

```mysql
# if
if 条件表达式 then
	--SQL
else
	--SQL
end if;

# case
case 变量
when 值	then
	--SQL
when 值	then
	--SQL
...
else
	--SQL(如果变量的值全部不匹配)
end case;
```

#### 循环语句

```mysql
# while
declare i int default 1;
while 条件表达式 do
	-- 	SQL
	set i = i + 1;
end while;

# repeat
declare i int default 1;
repeat 
	-- 	SQL
	set i = i + 1;
until 条件表达式
end repeat;

# loop
declare i int default 1;
set i = 1;
循环名: loop
	--SQL
	if 条件表达式 then
		leave 循环名
	end if;
end loop;
```

### 存储过程管理

存储过程是隶属于某个数据库的，只能在当前数据库中使用

#### 查询存储过程

```mysql
# 查询[某个数据库]所有的存储过程
show procedure status [where db = 某个数据库];
# 查询存储过程的创建细节
show create procedure 数据库.存储过程名;
```

#### 修改存储过程 alter

```mysql
# 主要是修改存储过程的特征/特性
alter procedure <存储过程名> 特征1 [特征2 特征3 ...]
```

#### 删除存储过程 drop

```mysql
drop procedure
```

### 游标

```mysql
# 对于返回的多条sql信息，可以使用游标实现结果集的遍历
# 游标	——	  类比于迭代器

# 步骤
# 1. 申明游标
declare cursor_name cursor for DQL;
# 2. 打开游标
open cursor_name;
# 3. 使用游标要使用循环语句
declare n int;
declare i int default 0;
declare str varchar(50);

select count(*) INTO n from 表;
while i < n do
	# 4. 使用游标,提取游标当前指向的记录(提取之后，游标自动下移)
	FETCH cursor_name INTO 列1 [列2, 列3 ...]
	set str = concat_ws(' ',列1 [列2, 列3 ...]);
	set 输出变量 = concat_ws('\n',输出变量,str);
	set i = i + 1;
end while;
# 5. 关闭游标
close cursor_name;
```

## 触发器

触发器和存储过程一样是一个能完成特定功能，存储在数据库服务器上的SQL片段。

但是触发器无需调用，当对数据表中的数据执行DML操作时自动触发这个SQL片段执行，无需手动调用。

在MySQL中，只有执行DML操作的时候才会触发触发器的执行。

### 创建触发器	create trigger

```mysql
# 语法
create trigger <tri_name>
before | after 					-- 定义触发时机
<insert | delete | update>		-- 定义DML类型
ON <table_name>					-- 确定表名
for each row					-- 申明为行级触发器（只要操作一条记录就会触发触发器一次）
begin
	-- SQL 触发器操作
end;


对于需要报错的地方，用下面这句话实现：
DECLARE msg VARCHAR(100);
SIGNAL SQLSTATE 'HY000' SET message_text = msg;
```

### 查询触发器	show triggers

```mysql
show triggers；
```

### 删除触发器	drop trigger

```mysql
drop trigger <tri_name>;
```

### NEW | OLD

```mysql
-- 我们可以使用NEW | OLD 在触发器中获取触发这个触发器的DML操作的数据
-- NEW：在触发器中用户获取insert操作添加的数据，update操作修改后的记录
-- OLD：在触发器中用于获取delete操作删除前的数据，update操作修改前的数据

```

### 触发器优点

```mysql
-- 可以实现表中数据的级联操作
-- 可以对DML操作的数据进行合法性校验
```



## 索引

索引，就是将数据表中某一列/某几列的值取出来构造便于查找的结构进行存储，生成数据表的**目录**。

当我们进行数据查询的时候，则先在**目录**中进行查找得到对应的数据的地址，然后再到数据表中根据地址快速的获取数据记录，避免全表扫描。

### 索引的分类

```mysql
-- MySQL中的索引，根据创建索引的列的不同，可以分为：

- 主键索引：在数据表的主键字段创建的索引，这个字段必须被 primary key修饰，每张表只能有一个主键
- 唯一索引：在数据表中的唯一列创建的索引(unique)，此列的所有制只能出现一次，可为NULL
- 普通索引：在普通字段上创建的索引，没有唯一性的限制
- 组合索引：两个及其以上字段联合起来创建索引的方式

-- 说明：
1. 在创建数据表时，将字段声明为主键（添加主键约束），会自动在主键字段创建主键索引
2. 在创建数据表时，将字段声明为唯一键（添加唯一约束），会自动在唯一字段创建唯一索引
```

### 创建索引

```mysql
create index <索引名>
on <表名>(列名);

-- 创建唯一索引：创建唯一索引的列的值不能重复
create unique index <索引名> on  <表名>(列名);

-- 创建普通索引：不要求创建索引的列的值的唯一性
create index <索引名> on  <表名>(列名);

-- 创建组合索引：
create index <索引名> on  <表名>(列名1,列名2);
```

### 使用索引

```mysql
-- 索引创建完毕之后无需调用，当根据创建索引的列进行数据查询的时候，会自动使用索引
-- 组合索引需要根据创建索引的所有字段进行查询时触发
```

### 查看索引

```mysql
show indexes from table <表名>;
show keys from table <表名>;
```

### 删除索引

```mysql
drop index <索引名> on 表名;
```







## 常见函数

```mysql
# 语法
select 函数名(实参列表) 【from 表】;

# 分类
1. 单行函数
	如 concat,length,ifnull等
2. 分组函数
	功能：做统计使用，称为聚合函数
```

### 字符函数

```mysql
# length(str)					获取参数值的字节个数

# concat(str1,str2)				拼接字符串

# upper/lower(str)				大写/小写字符串

# substr/substring(str,pos,len)	截取pos后len个字符(默认到末尾)的字符串(索引从1开始)

# instr(str1,str2)				在str1中搜索str2第一次出现的位置(索引从1开始)，若不存在返回0

# trim(‘ch’ from str)			去除字符串前后多余的字符ch(默认空格)

# lpad(str1, len, str2)			用str2实现左填充str1至指定长度，如果len(str1) > len, 则从str1左侧开始保留len个字符

# rpad(str1, len, str2)			用str2实现右填充str1至指定长度，如果len(str1) > len, 则从str1左侧开始保留len个字符

# replace(str,str1,str2)		用str2全部替换str中的str1
```



### 数学函数

```mysql
# round(x,D) 					四舍五入浮点数x保留至小数点后D位	

# ceil(x)						向上取整

# floor(x)						向下取整

# truncate(x,D)					截断，即阶段浮点数x保留至小数点后D位

# mod(a,b)						取余

```



### 日期函数

```mysql
# now()						当前时间，年月日时分秒
# curdate()					当前年月日
# curtime()					当前时分秒
# year/month(上面三个)		 取时间的具体部分

# str_to_date				字符串变为日期格式
# date_format				将日期转换为字符串

# datediff(日期1, 日期2)	  返回两个日期之间的天数
```



### 流程控制函数

```mysql
# if(expr1, expr2, expr3)	类似三目运算符，如果expr1成立，输出expr2,否则expr3

# case  一，实现 switch - case 的效果
	case 要判断的字段或者表达式
	when 常量1 then 要显示的值1或语句1;
	when 常量2 then 要显示的值2或语句2;
	...
	else 要显示的值n或者语句n;
	end
	
# case	二，实现 多级if 的效果
	case
	when 常量1 then 要显示的值1或语句1;
	when 常量2 then 要显示的值2或语句2;
	...
	else 要显示的值n或者语句n;
	end
```



### 分组函数（聚合函数）

- 功能：用作统计使用，又称为聚合函数或统计函数或组函数
- 分类：sum , avg , mas , min , **count** 

```mysql
# sum, avg 					一般用于处理数值型
# max,min,count 			可以处理任何类型

# 以上分组函数均忽略null值，而且都可以进行distinct的去重处理
select avg(distinct salary) from employees;

# count 函数的详细介绍——可以使用 count(*) 或 count(常量) 来统计所有行数
select count(*) from employees;
select count(1) from employees;

# 和分组函数一同查询的字段有限制(和分组函数一同查询的字段要求是group by后的字段)
```





