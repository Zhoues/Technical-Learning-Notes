# Spring 5

[TOC]



# Spring5框架概述

- Spring是轻量级的开源的JavaEE框架

- Spring可以解决企业开发的复杂性

- Spring有两个核心的部分：IOC 和 AOP
  - IOC：控制反转，把创建对象过程交给Spring进行管理
  - AOP：面向切面，不修改源代码进行功能的增强

- Spring特点
  - 方便解耦，简化开发
  - AOP变成支持
  - 方便程序的测试
  - 方便和其他框架进行整合
  - 方便进行事务的操作
  - 降低API的开发难度



# IOC容器

## IOC基本概念

IOC（Inversion of Control）称为控制反转，是面向对象编程的一种设计原则。

IOC把对象创建和对象之间的调用过程交给Spring进行管理

使用IOC的目的是为了降低耦合度

## IOC底层原理

1. xml解析
2. 工厂模式
3. 反射

![图1](C:\Users\这是恩申的哟\Desktop\Technical-Learning-Notes\SSM\Spring\picture\图1.png)

![图2](C:\Users\这是恩申的哟\Desktop\Technical-Learning-Notes\SSM\Spring\picture\图2.png)

## IOC接口（BeanFactory,ApplicationContext）

- IOC思想基于IOC容器完成，IOC容器底层就是对象工厂
- Spring提供IOC容器实现的两种方式：（两个接口）
  - BeanFactory：IOC容器基本实现，是Spring内部使用的接口，不提供开发人员进行使用
    - 加载配置文件时不会创建对象，在获取对象（使用对象）的时候才会创建对象
  - ApplicationContext：BeanFactory接口的子接口，提供更多更强大的功能，一般面向开发人员使用
    - 加载配置文件的时候就会把在配置文件中的对象进行创建

- ApplicationContext有主要的实现类
  - ClassPathXmlApplicationContext：类相对路径
  - FileSystemXmlApplicationContext：盘绝对路径

## IOC操作Bean管理（基于XML）

什么是Bean管理，Bean管理指的是两个操作

- Spring创建对象
- Spring注入属性



### 一、基于XML创建对象

```xml
<!--    配置User对象创建      -->
<bean id="user" class="com.example.spring5.User"></bean>
```

在spring配置文件中，使用bean标签，标签里面添加对应属性，就可以实现对象创建

在bean有很多属性：

- **id：注册类是分配的唯一标识**
- class：类全路径（包内路径）

创建对象时候，默认也是执行无参构造方法



### 二、基于XML注入属性

DI：依赖注入，就是输入属性



#### 注入一般类型属性



**第一种注入方式：使用set方法进行注入**

- 创建类，定义属性和对应的Set方法
- 在spring配置文件配置对象创建，配置属性注入

```xml
<bean id="book" class="">
	<!--使用 property 完成属性注入
		name：类里面的属性名称
		value：想属性注入的值
	-->
    <property name="bname" value="易筋经"></property>
</bean>
```



**第二种注入方式：使用有参数构造进行注入**

- 创建类，定义属性和对应的有参构造方法
- 在spring配置文件配置对象创建，配置属性注入

```xml
<bean id="book" class="">
	<!--使用 constructor-arg 完成属性注入
		name：类里面的属性名称
		value：想属性注入的值
		index：有参构造的第几个参数
	-->
    <constructor-arg name="bname" value="易筋经"></constructor-arg>
    <constructor-arg index="1" value="14"></constructor-arg>
</bean>
```



**第三种注入方式：p名称空间注入（简化set方法）**

- 使用 p 名称空间注入，可以简化基于xml配置方式

- 添加p名称空间的配置
```xml
  <?xml version="1.0" encoding="UTF-8"?>
  <beans xmlns="http://www.springframework.org/schema/beans"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xmlns:p="http://www.springframework.org/schema/p"
         xsi:schemaLocation="http://www.springframework.org/schema/beans http://www.springframework.org/schema/beans/spring-beans.xsd">
  </beans>
```

- 进行属性，在bean标签

```xml
<bean id="book" class="" p:bame="九阳神功" p:bauthor="无名氏"></bean>
```



#### 注入特殊类型属性

字面量

- null值

```xml
<bean id="book" class="">
    <property name="address">
    	<null/>
    </property>
</bean>
```



- 属性值包含特殊符号
  - 用html中的大于和小于号进行转义 \&lt; \&gt;
  - 把特殊符号内容写到CDATA

```xml
<bean id="book" class="">
    <property name="address"> <![DATA[ <<南京>> ]]>  </property>
</bean>
```



#### 注入外部bean （对于外部bean采用set方法进行注入）

```xml
<bean id="userService" class="">
	<!--使用 property 完成属性注入
		name：类里面的属性名称
		ref：创建userDao对象bean标签的id值
	-->
    <property name="userDao" ref="userDaoImpl"></property>
</bean>

<bean id="userDaoImpl" class=""></bean>
```



#### 注入内部bean

一对多：部门与员工

```xml
<bean id="employee" class="">
    <property name="department">
    	<bean id="employee" class="">
        	<property name="dname" ref="安保部门"></property>
        </bean>
    </property>
</bean>
```



#### 注入级联赋值（和注入外部bean或者采用get方法级联赋值）

```xml
<bean id="employee" class="">
    <!--注入外部bean-->
    <property name="department" ref="employee"></property>
    <!--采用get方法级联赋值-->
    <property name="department.dname" value="安保部门"></property>	
</bean>
<bean id="employee" class="">
    <property name="dname" ref="安保部门"></property>
</bean>
```



#### 注入集合属性（集合内部是基本数据类型）

数组，List，Map

```xml
<bean id="stu" class="">
    <property name="array">
    	<array>
        	<value>java课程</value>
            <value>数据结构课程</value>
        </array>
    </property>
    
    <property name="list">
    	<list>
        	<value>java课程</value>
            <value>数据结构课程</value>
        </list>
    </property>
    
    <property name="map">
    	<map>
        	<entry key="" value=""></entry>
            <entry key="" value=""></entry>
        </map>
    </property>
    
    <property name="set">
    	<set>
        	<value>java课程</value>
            <value>数据结构课程</value>
        </set>
    </property>
</bean>
```



#### 注入集合属性（集合内部是对象）

```xml
<bean id="stu" class="">
    <property name="list">
    	<list>
            <ref bean="course1"></value>
            <ref bean="course2"></value>
        </list>
    </property>
</bean>
```







## IOC操作Bean管理（基于注解）
