# SpringBoot2

[TOC]

# 创建SpringBoot2

## 创建项目

1. 使用IDEA自带的SpringBoot Initializr  创建
2. 使用`start.spring.io`
3. 使用阿里云镜像创建 `http://start.aliyun.com`替换`http://spring.io`
4. 创建Maven项目



# SpringBoot2组成

```apl
SpringBoot
|
|-.idea
|-.mvn
|- src
	|
	| - java
		  | - SpringBootApplication (SpringBoot项目启动入口类)
	| - resources
    	  | - static (静态资源)
    	  | - templates (页面模板)
    | - application.properties (核心配置文件)
| - pom.xml (核心依赖)
```

## pox.xml

```xml
<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 https://maven.apache.org/xsd/maven-4.0.0.xsd">
    <modelVersion>4.0.0</modelVersion>
    <!--  SpringBoot父工程GAV  -->
    <parent>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-parent</artifactId>
        <version>2.7.0</version>
        <relativePath/> <!-- lookup parent from repository -->
    </parent>

    <!--  当前项目GAV  -->
    <groupId>com.example</groupId>
    <artifactId>SpringBootDemo</artifactId>
    <version>0.0.1-SNAPSHOT</version>

    <name>SpringBootDemo</name>
    <description>SpringBootDemo</description>

    <!--  编译级别  -->
    <properties>
        <java.version>17</java.version>
    </properties>

    <!--  依赖  -->
    <dependencies>
        <!--  SpringBoot框架web项目起步依赖  -->
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-web</artifactId>
        </dependency>
        <!--  SpringBoot框架测试起步依赖  -->
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-test</artifactId>
            <scope>test</scope>
        </dependency>
    </dependencies>

    <build>
        <plugins>
            <!--  SpringBoot编译打包项目插件  -->
            <plugin>
                <groupId>org.springframework.boot</groupId>
                <artifactId>spring-boot-maven-plugin</artifactId>
            </plugin>
        </plugins>
    </build>

</project>
```

### starter

- SpringBoot 中常见的项目名称，定义了当前项目使用的所有以来坐标，以达到**减少依赖配置**的目的

### parent

- 所有Springboot项目要继承的项目，定义了若干个坐标版本号（依赖管理，而非依赖），以达到**减少依赖冲突**的目的
- spring-boot-starter-parent各版本之间存在着诸多坐标版本不同

### 实际开发

- 使用任意坐标时，仅书写GAV中的G和A，V由SpringBoot提供
- 如果坐标错误，再指定Version(**小心版本冲突**)

## SpringBootApplication

```java
package com.example.springbootdemo;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

// SpringBoot项目启动入口类
@SpringBootApplication      // Springboot核心组件，主要用于开启Spring自动配置
public class SpringBootDemoApplication {

    public static void main(String[] args) {
        // 启动一个SpringBoot容器
        SpringApplication.run(SpringBootDemoApplication.class, args);
    }
}
```

- SpringBoot的引导类是Boot工程的执行入口，允许main方法就可以启动项目
- SpringBoot工程运行后初始化Spring容器，扫描引导类所在的包加载bean

## application.*

加载顺序：properties > yml > yaml

### properties

```yaml
# 设置内嵌Tomcat端口号
server.port=8000

# 修改banner
spring.main.banner-mode=off	# 直接取消banner
spring.banner.image.location=logo.png	# 修改banner的样式图案

# 日志
logging.level.root=info/error/debug		# 终端控制台显示的日志信息

# 设置上下文根
server.servlet.context-path=/api
```

### yml(主流格式)

- 大小写敏感
- 属性层级关系使用多行描述，每行结尾使用冒号结束
- 使用缩进表示层级关系，同层级左侧对齐，只允许使用空格
- **属性值前面添加空格（属性名与属性值之间使用冒号+空格作为分隔）**
- \# 表示注释



#### 用户自定义参数		单个获取@Value

在配置文件里面可以添加用户自动的变量，但是在别的文件使用的时候需要采用特殊的语法

- 使用@Value读取单个数据，属性名引用方式`${一级属性名.二级属性名}`

```yaml
./application.properties
website = www.super2021.com
# 使用${属性名}引用数据
website_sub = ${website}/api   
# 使用引号包裹的字符串，其中的转义字符可以生成
user:
	name: Zhoues
        
likes:
	- music
    - art

users:
	-
        name: Zhoues
        age: 18
    - 
        name: Super2021
        age: 20
            
./controller/IndexController

@Value("${website}")
private String website;

@Value("${user.name}")
private String name;

@Value("${like[0]}")
private String subject;

@Value("${users[0].name}")
private String name;
```

#### 用户自定义参数		全部获取@Autowired

```java
application.yml内容同上
@Autowired
private Environment env;
env.getProperty("website");
env.getProperty("user.name");
```

#### 用户自定义参数	 	获取引用类型属性数据

- 创建类
- 由Spring帮我们去加载数据到对象中，一定要告诉spring加载这组信息
- 使用时候从Spring中直接获取信息使用

```yaml
datasource:
	driver: web
	url: www.super2021.com
```

```java
// 1. 定义数据模型封装 yanml文件中对应的数据
// 2. 定义为spring 管控的 bean
@Component
// 3. 指定加载的数据
@ConfigurationProperties(prefix = 'datasource')
```



### 多环境多配置文件	

比如需要多个环境采取多个不一样的配置，我们可以设置多个application.properties，命名要求如下：

`application-环境.properties`

此时主配置文件需要指定环境：

```yaml
# 主核心配置文件中可以激活使用的配置文件
spring.profiles.active=环境
```

# REST开发

根据REST风格对资源进行访问称为RESTful

开发样例

```java
package com.example.springbootdemo.Controller;

import com.example.springbootdemo.Entity.User;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.*;

/* REST风格开发接口 */
//  @Controller
//  @ResponseBody
@RestController
@RequestMapping("/users")
public class UserController {

    //    @RequestMapping(method = RequestMethod.POST)
    @PostMapping
    public String save(@RequestBody User user) {
        System.out.println("user save...");
        return "user save...";
    }

    //    @RequestMapping(method = RequestMethod.PUT)
    @PutMapping
    public String update(@RequestBody User user) {
        System.out.println("user update..." + user);
        return "user update..." + user;
    }

    //    @RequestMapping(value = "/{id}", method = RequestMethod.DELETE)
    @DeleteMapping("/{id}")
    public String delete(@PathVariable Integer id) {
        System.out.println("user delete..." + id);
        return "user delete..." + id;
    }

    //    @RequestMapping(value = "/{id}", method = RequestMethod.GET)
    @GetMapping("/{id}")
    public String getById(@PathVariable Integer id) {
        System.out.println("user get by ..." + id);
        return "user get by..." + id;
    }

    //    @RequestMapping(method = RequestMethod.GET)
    @GetMapping
    public String getALL() {
        System.out.println("user get all ...");
        return "user get...";
    }
}

```

@RequestBody		@RequestParam		@PathVariable

- 区别
  - @RequestParam 用来接收url地址传参或表单传参
  - @RequestBody	用于接收json数据
  - @PathVariable     用于接收路径参数，使用{参数名称}描述路径参数
- 应用
  - 后期开发，发送请求参数超过一个，以json个数为主，@RequestBody应用广泛
  - 如果发送非json格式数据，选用@RequestParam接收请求参数
  - 采用RESTful进行开发，当参数数量较少，例如一个，可以采用@PathVariable接收请求路径变量，通常用户传递id值

# 整合第三方技术

- 导入对应的starter
- 配置对应的设置或采用默认的配置

## 整合Junit

在SpringBootTest注解下的类进行测试操作即可

```java
@SpringBootTest
class SpringBootDemoApplicationTests {

    @Test
    void contextLoads() {
    }

}
```

## 整合mybatis

pom.xml

```xml
<dependency>
    <groupId>mysql</groupId>
    <artifactId>mysql-connector-java</artifactId>
    <version>8.0.25</version>
</dependency>
<dependency>
    <groupId>org.mybatis.spring.boot</groupId>
    <artifactId>mybatis-spring-boot-starter</artifactId>
    <version>2.2.0</version>
</dependency>
```

application.yml

```yml
spring:
  datasource:
    driver-class-name: com.mysql.jdbc.Driver
    url: 
    username: 
    password: 
mybatis:
  mapper-locations: classpath:Mapper/*xml
  type-aliases-package: com.example.springboottest.Entity

```

之后的操作基本一致

## 整合mybatis-plus

在Dao层可以直接继承大量API

```java
@Mapper
public interface BookDao extends BaseMapper<Book>{
    
}
```

application.yml

```yml
mybatis-plus:
	global-config:
		db-config:
			table-prefix: tbl_
			id-typd: auto
	# 打开日志
	configuration:
		log-impl: org.apache.ibatis.logging.stdout.StdOutImpl
```



## 整合lombok

```xml
<dependency>
	<groupID>org.projectlombok</groupID>
    <artifactId>lombok</artifactId>
</dependency>
```

可以直接添加注解完成实体类的快速开发

```java
@Data
@AllArgsConstructor
@NoArgsConstructor
public User(){
    
}
```



# 快速开发SpringBoot

## 数据层使用MP快速开发

```java
public interface BookDao extends BaseMapper<Book>{
    // 自己的查询方法
}
```

## 业务层使用MP快速开发

```java
public interface IbookService extends IService<Book>{
    
}
```

```java
public class BookServiceImpl extends ServiceImpl<BookDao,Book> implementsIbookService{
    
}
```



## 统一Controller返回信息

定义一个ReturnMessage类，主要包括是否成功，返回信息和返回数据

## 异常处理

在Controller中的utils工具包中添加异常处理方法

```java
// @ControllerAdvice
@RestControllerAdvice
public class ProjectExceptionAdvice{
    // 拦截所有的异常信息
    @ExceptionHandler
    public ReturnMessage doException(Exception ex){
        // 记录日志
        // 通知运维
        // 通知开发
        ex.printStackTrance();
        return ReturnMessage(...)
    }
}
```



# 打包与运行

- 对SpringBoot项目打包（指向Maven构建指令package）
  - 要么`mvn package`
  - 要么直接双击IDEA右侧Maven中的package指令
- 运行项目（指向启动指令）
  - `java -jar springboot.jar`
  - jar支持命令行启动需要依赖maven插件支持，确认打包时是否具有SpringBoot对应的maven插件

```xml
    <build>
        <plugins>
            <!--  SpringBoot编译打包项目插件  -->
            <plugin>
                <groupId>org.springframework.boot</groupId>
                <artifactId>spring-boot-maven-plugin</artifactId>
            </plugin>
        </plugins>
    </build>
```



# 配置高级

## 临时属性

- 使用jar命令启动SpringBoot工程可以使用临时属性替代配置文件中的属性
- 临时属性添加方式：`java -jar 工程名.jar --属性名=值`
- 多个临时属性之间使用空格分隔
- 临时属性必须时当前boot工程支持的属性，否则设置无效

## 开发环境配置临时属性

- 带属性启动SpringBoot程序，为程序添加运行属性
- 通过编程形式带参数启动SpringBoot程序，为程序添加运行参数

```java
// SpringBoot项目启动入口类
@SpringBootApplication      // Springboot核心组件，主要用于开启Spring自动配置
public class SpringBootDemoApplication {

    public static void main(String[] args) {
        String[] arg = new String[1];
        arg[0] = "--server.port=8080";
        SpringApplication.run(SpringBootDemoApplication.class, arg);
    }
}
```

- 不携带参数启动SpringBoot程序

```java
// SpringBoot项目启动入口类
@SpringBootApplication      // Springboot核心组件，主要用于开启Spring自动配置
public class SpringBootDemoApplication {

    public static void main(String[] args) {
        SpringApplication.run(SpringBootDemoApplication.class);
    }
}
```

## 配置文件分类

### SpringBoot中4级配置文件

- 1级：file : config/application.yml                             【最高】
- 2级：file : application.yml
- 3级：classpath: config/application.yml
- 4级：classpath: application.yml                                【最低】



- 1级和2级时留做系统打包之后设置通用属性，1级常用于运维经理进行线上整体项目部署方案调控
- 3级和4级用于系统开发设置通用属性，3级常用于项目经理进行整体项目的属性调控



- **多层级配置文件间的属性采用叠加并覆盖的形式作用与程序**

## 自定义配置文件

- 通过启动参数加载配置文件（无需写配置文件的扩展名）
  - `--spring.config.name=xxx`

- 通过启动参数加载指定文件路径下的配置文件时可以加载多个配置
  - `--spring.config.location=classpath:/.....`





















# 多环境开发

## 单文件yml版

- 使用`---`区分环境设置的边界
- 每一种环境的区别在于加载的配置属性不同
- 启动某种环境时需要指定启动使用该环境

```yml
# 应用环境
# 公共配置

spring:
  profiles:
    active: pro

---
# 生产环境
spring:
  config:
    activate:
      on-profile: pro

---
# 开发环境
spring:
  config:
    activate:
      on-profile: dev

---
# 测试环境
spring:
  config:
    activate:
      on-profile: test

```

## 多文件yml版

- 主启动配置文件`application.yml`
- 环境分类配置文件`application-pro.yml`,`application-dev.yml`,`application-test.yml`



## 多环境开发分组管理

- 根据**功能**对配置文件中的信息进行拆分，并制作乘独立的配置文件，命名规则如下
  - `application-devDB.yml`
  - `application-devRedis.yml`
  - `application-devMVC.yml`

- 使用**include属性**在激活指定环境的情况下，同时对多个环境进行加载使其生效，多个环境间使用逗号分隔
- 当主环境dev与其他环境有相同属性时，**主环境属性生效**；其他环境中有相同属性时，**最后加载的环境属性生效**

```yaml
spring:
  profiles:
    active: dev
    include: devDB, devRedis, devMVC
```

- 使用group属性替代include属性，降低了属性配置量，可以使用group属性定义多个主环境与子环境的包含关系

```yaml
spring:
  profiles:
    active: dev
    group: 
      "dev": devDB, devRedis, devMVC
      "pro": proDB, proRedis, proMVC
```



# 日志

## 日志的基本操作

在代码中使用日志工具记录日志

```java
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/* REST风格开发接口 */
//  @Controller
//  @ResponseBody
@RestController
@RequestMapping("/users")
public class UserController {
    private static final Logger log = LoggerFactory.getLogger((UserController.class));
    //    @RequestMapping(method = RequestMethod.POST)
    @PostMapping
    public String save(@RequestBody User user) {
        System.out.println("user save...");
        log.debug("debug...");
        log.info("info...");
        log.warn("warn...");
        log.error("error...");
        return "user save...";
    }


```

设置日志输出级别

```yaml
# 开启 debug 模式，输出调试信息，常用于检查系统运行状况
debug: true
# 设置日志级别，root表示根节点，即整体应用日志级别
logging:
  # 设置分组，对某个组设置日志记录
  group:
    dev: com.example.controller,com.example.service,com.example.dao
    iservice: com.alibaba
  level:
    root: info
    # 设置某个包的日志级别
    com.example.controller: debug
    # 设置分组的日志级别
    dev: warn
```

## 优化日志对象的创建

使用lombok提供的助教@Slf4j简化开发，减少日志对象的声明操作

```java
@Slf4j
@RestController
@RequestMapping("/users")
public class UserController {
    @PostMapping
    public String save(@RequestBody User user) {
        System.out.println("user save...");
        log.debug("debug...");
        log.info("info...");
        log.warn("warn...");
        log.error("error...");
        return "user save...";
    }
```



# 热部署

## 手动热部署

- 添加开发者工具依赖

```xml
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-devtools</artifactId>
</dependency>
```

- 激活热部署：Ctrl + F9
- 关于热部署：
  - 重启：自定义开发代码，包含类、页面、配置文件等，加载位置restart类加载器
  - 重载：jar包，加载位置base类加载器

## 自动启动热部署

- 设置自动构建项目
  - settings => Build,Exeception,Deplyment => Compiler => Build project automatically √
- 设置自动构建项目
  - `Ctrl + Alt + Shift + /`打开注册表勾选

## 设置热部署触发的范围

- 默认不触发重启的目录列表
  - /META-INF/maven
  - /META-INF/resources
  - /resources
  - /static
  - /public
  - /templates
- 手动设置排除项

```yaml
spring:
  devtools:
    restart:
      exclude: static/**
```

## 禁用热部署

可以直接在yml中使用enabled为false

```yaml
spring:
  devtools:
    restart:
      exclude: static/**
      enabled: false
```

当然也可以直接在更高基本的设置上设置禁用

```java
// SpringBoot项目启动入口类
@SpringBootApplication      // Springboot核心组件，主要用于开启Spring自动配置
public class SpringBootDemoApplication {
    public static void main(String[] args) {
        System.setProperty("spring.devtools.restart.enabled", "false");
        SpringApplication.run(SpringBootDemoApplication.class, args);
    }
}
```

