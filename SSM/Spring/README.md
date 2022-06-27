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

![图1](\picture\图1.png)

![图2](\picture\图2.png)

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
    <property name="dname" value="安保部门"></property>
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



### 三、将集合注入部分提取出来

**采用util命名空间法**

- 在spring配置文件中引入名称空间util

```xml
  <?xml version="1.0" encoding="UTF-8"?>
  <beans xmlns="http://www.springframework.org/schema/beans"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xmlns:util="http://www.springframework.org/schema/util"
         xsi:schemaLocation="http://www.springframework.org/schema/beans    http://www.springframework.org/schema/beans/spring-beans.xsd http://www.springframework.org/schema/util http://www.springframework.org/schema/util/spring-util.xsd">
  </beans>
```

- 使用util标签完成list集合进行注入

```xml
<!-- 1. 提取list集合类型属性进行注入 -->
<util:list id="bookList">
	<value>java课程</value>
    <value>数据结构课程</value>
</util:list>

<!-- 2. 提取list集合类型属性注入使用 -->
<bean id="book" class="">
	<property name="list" ref="bookList"></property>
</bean>
```



## IOC操作Bean管理（FactoryBean）

1. Spring有两种类型的bean，一种普通bean，另一种工厂bean(FactoryBean)
2. 普通bean：在配置文件中，定义bean的类型就是返回类型
3. 工厂bean：在配置文件中，定义bean的类型可以和返回类型不一样
   1. 创建类，让这个类作为工厂bean，实现接口 FactoryBean
   2. 实现接口里面的方法，在实现的方法中定义返回的bean类型

```java
public class MyBean implements FactoryBean {
    // 这个时候可以定义返回值类型，此时的返回值的类型就是工厂Bean生产出来的类型，不一定和包管理的类一致
    @Override
    public Object getObject() throws Exception {
        return null;
    }

    @Override
    public Class<?> getObjectType() {
        return null;
    }

    @Override
    public boolean isSingleton() {
        return FactoryBean.super.isSingleton();
    }
}
```

## IOC操作Bean管理（Bean的作用域）

1. 在Spring里面，可以创建bean实例是单实例还是多实例
2. 在Spring里面，默认情况下，bean是单实例对象



**如何设置单实例还是多实例**

1. 在spring配置文件bean标签里面有属性(scope)用于设置单实例还是多实例
2. scope属性值

第一个值  默认值，singleton，表示单实例对象

第二个值 prototype，表示的是多实例对象。设置scope的值是prototype的时候，不是加载spring配置文件时候创建对象，在调用getBean方法时候创建多实例对象



## IOC操作Bean管理（Bean的生命周期）

生命周期：从对象创建到对象销毁的过程



**bean 生命周期**

1. 通过构造器创建bean实例（无参数构造）
2. 为bean的属性设置值和对其他bean引用（调用set方法）
3. 调用bean的初始化的方法（需要进行配置）
4. bean可以使用了（对象获取到了）
5. 当容器关闭的时候，调用bean的销毁的方法（需要进行配置销毁的方法）

```xml
<bean id="orders" class="" init-method="initMethod" destroy-method="dstroyMethod">
	<property name="oname" value="手机"></property>
</bean>
```



如果添加bean后置处理器有七步

1. 通过构造器创建bean实例（无参数构造）
2. 为bean的属性设置值和对其他bean引用（调用set方法）
3. 把bean实例传递给bean后置处理器的方法——postProcessBeforeIntialization
4. 调用bean的初始化的方法（需要进行配置）
5. 把bean实例传递给bean后置处理器的方法——postProcessAfterIntialization
6. bean可以使用了（对象获取到了）
7. 当容器关闭的时候，调用bean的销毁的方法（需要进行配置销毁的方法）

实现java后置处理器接口

```java
import org.springframework.beans.factory.config.BeanPostProcessor;

public class MyBeanPost implements BeanPostProcessor {

    @Override
    public java.lang.Object postProcessBeforeInitialization(java.lang.Object bean, java.lang.String beanName) throws org.springframework.beans.BeansException { 
        return null;
    }
    @Override
    public java.lang.Object postProcessAfterInitialization(java.lang.Object bean, java.lang.String beanName) throws org.springframework.beans.BeansException { 
        return null;
    }
}
```

## IOC操作Bean管理（xml自动注入）

自动装配：根据指定装配规则（属性名称或者属性类型），Spring自动将匹配属性值进行注入

实现自动装配：

1. bean标签属性autowire，配置自动装配
2. autowire属性常用的两个值
   1. byName 根据属性名称注入，注入值bean的id值和类属性名称一样
   2. byType根据属性类型注入，



## IOC操作Bean管理（外部属性文件）

- 在spring配置文件中引入名称空间context

```xml
  <?xml version="1.0" encoding="UTF-8"?>
  <beans xmlns="http://www.springframework.org/schema/beans"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xmlns:util="http://www.springframework.org/schema/util"
         xmlns:context="http://www.springframework.org/schema/context"
         xsi:schemaLocation="http://www.springframework.org/schema/beans    http://www.springframework.org/schema/beans/spring-beans.xsd http://www.springframework.org/schema/util http://www.springframework.org/schema/util/spring-util.xsd
http://www.springframework.org/schema/context http://www.springframework.org/schema/context/spring-context.xsd" >
  </beans>
```

- 在spring配置文件中引入外部属性配置文件

```xml
<context:property-placeholder location="classpath：jdbc.proerties"/>
```

- 用`${}`引入即可



## IOC操作Bean管理（基于注解）

什么是注解

1. 注解是代码特殊标记，格式：`@注解名称(属性名称=属性值,属性名称=属性值...)`
2. 注解可以作用在类上面，方法甚至属性上面
3. 使用注解目的：简化xml配置



Spring针对Bean管理中创建对象提供注解

1. `@Component`
2. `@Service`
3. `@Controller`
4. `Repository`

上面四个注解的功能都是一样的，都可以用来创建bean实例



### 一、基于注解创建对象

1. 引入aop依赖
2. 开启组件扫描：引入context空间

```xml
  <?xml version="1.0" encoding="UTF-8"?>
  <beans xmlns="http://www.springframework.org/schema/beans"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xmlns:util="http://www.springframework.org/schema/util"
         xmlns:context="http://www.springframework.org/schema/context"
         xsi:schemaLocation="http://www.springframework.org/schema/beans    http://www.springframework.org/schema/beans/spring-beans.xsd http://www.springframework.org/schema/util http://www.springframework.org/schema/util/spring-util.xsd
http://www.springframework.org/schema/context http://www.springframework.org/schema/context/spring-context.xsd" >
      
      <!--开启组件扫描
			1. 如果扫描多个包，多个包使用逗号隔开
			2. 扫描包上层目录
		-->
      <context:component-scan base-package=""></context:component-scan>
</beans>
```

3. 创建类，在类上创建注解

```java
package com.example.spring5.service;

import org.springframework.stereotype.Component;

// 在注解里面value属性值可以省略不写
// 默认值是类名称，首字母小写
// UserService -- userService
@Component(value = "userService")   // <bean id="userService" class=".."/>
public class UserService {

}
```

4. 开启组件扫描细节配置

```xml
<!-- 实例1
 	use-default-filters="false"：表示现在不使用默认的filter，自己配置filter
	context:include-filter：设置扫描哪些内容
		-->
<context:component-scan base-package="" use-default-filters="false">
    <context:include-filter type="annotation" expression="org.springframework.stereotype.Controller"/>
</context:component-scan>

<!-- 实例1
	context:exclude-filter：设置不扫描哪些内容
		-->
<context:component-scan base-package="">
    <context:exclude-filter type="annotation" expression="org.springframework.stereotype.Controller"/>
</context:component-scan>
```

### 二、基于注解方式实现属性注入

**@Autowired：根据属性类型进行自动装配**

1. 把service和dao对象创建，在service和dao类添加创建对象的注解
1. 在service注入dao对象，在service类添加dao类型属性，在属性上面使用@Autowired注解

@Qualifier：根据属性名称进行注入

1. 一般和@Autowired一起使用（因为@Autowired是根据类型进行自动装配，但是如果有多个实现类会报错）
2. 根据名称（可以自定义）进行注入

@Resource：根据类型注入也可以根据名称注入

1. 单独的@Resource表示类型注入
2. @Resource(name="")表示名称注入

@Value：注入普通类型属性

```java
@Value(value="abc")
private String name;
```



## 三、完全注解开发

1. 创建配置类，替代xml配置文件

```java
@Configuration //作为配置类
@ComponentScan(basePackages={"包路径"})
public class SpringConfig{
    
}
```

2. 编写配置类

```java
@Test
public void testService(){
    ApplicationContext context 
        = new AnnotationConfigApplicationContext(SpringConfig.class);
}
```



# AOP

什么是AOP

1. 面向切面编程（面向方面编程），利用AOP可以对业务逻辑的各个部分进行隔离，从而使得业务逻辑各个部分之间的耦合度降低，提高程序的可重用性，同时提高了开发的效率
2. 



















