# ElasticSearch

[TOC]



## 下载ElasticSearch和Kibana

### ElasticSearch

网页

```
https://www.elastic.co/cn/downloads/past-releases/elasticsearch-7-17-0
```

#### 安装java jdk

```bash
#更新软件包列表
sudo apt-get update
#安装openjdk-17-jdk
sudo apt-get install openjdk-17-jdk
#查看版本
java -version
```

#### 下载安装包([ElasticSearch-7.17.0](https://www.elastic.co/cn/downloads/past-releases/elasticsearch-7-17-0))

```bash
#下载压缩包
wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.17.0-linux-x86_64.tar.gz
#解压
tar -zxvf elasticsearch-7.17.0-linux-x86_64.tar.gz -C /usr/local/
```

#### 修改elasticsearch.yml并开放防火墙端口

去腾讯云打开防火墙端口

```bash
vi /usr/local/elasticsearch-7.17.0/config/elasticsearch.yml
#修改network.host为 0.0.0.0
network.host: 0.0.0.0
#修改http.port为 9200
http.port: 9200
```

#### 新建(或使用已有)用户并赋予权限

```bash
#新建用户
adduser xxx
#已有用户
su xxx

#赋予权限
chown -R xxx /usr/local/elasticsearch-7.17.0/
```

#### root用户下更改内存权限

```bash
sysctl -w vm.max_map_count=262144
```

#### 修改jvm.option

```bash
vi /usr/local/elasticsearch-6.4.0/config/jvm.option 
#把1g改成4g
-Xms4g
-Xms4g
```

....

具体参考

https://blog.csdn.net/weixin_44596128/article/details/103970665

https://blog.csdn.net/fen_fen/article/details/123191203

https://blog.csdn.net/weixin_41698550/article/details/121837938



安装分词器

```bash
root@VM-24-14-ubuntu:/usr/local/elasticsearch-7.17.0/bin# ./elasticsearch-plugin install analysis-icu
```





```json
PUT /es_db
{
  "settings": {
    "index" : {
      "analysis.analyzer.default.type" : "ik_max_word"
    }
  }
}
```







