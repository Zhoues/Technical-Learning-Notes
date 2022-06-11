# Websocket

[TOC]



## 实时通信解决思路

**轮询**

让浏览器每隔一段时间向后台发一次请求。

缺点：延迟一段时间、请求太多，网站压力太大



**长轮询**

客户端向服务器发送请求，服务器会锁住请求并给请求一个最长等待时间，一旦有数据到来，就立即返回。**数据响应没有延迟**



**websocket**

客户端和服务器场景连接不断开，那么就可以先实现双向通信



## 轮询实现实时通信

### urls.py

```python
from django.contrib import admin
from django.urls import path
from app import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('home/', views.home),
    path('send/msg/', views.send_msg),
    path('get/msg/', views.get_msg),
]
```

### views.py

```python
from django.http import JsonResponse
from django.shortcuts import render, HttpResponse

# 一旦有新消息到来，我就放在数据库里
DB = []

def home(request):
    return render(request, 'home.html')


def send_msg(request):
    print("接收到客户端请求：", request.GET)
    text = request.GET.get('text')
    DB.append(text)
    return HttpResponse("OK")


def get_msg(request):
    index = request.GET.get('index')
    index = int(index)
    context = {
        "data": DB[index:],
        "max_index": len(DB)
    }
    return JsonResponse(context)
```

### home.html

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Title</title>
    <style>
        .message {
            height: 300px;
            border: 1px;
            width: 100%;
            background-color: bisque;
        }
    </style>
</head>
<body>
<div class="message" id="message"></div>
<div>
    <input type="text" placeholder="请输入" id="txt">
    <input type="button" value="发送" onclick="sendMessage();">
</div>
<script src="https://cdn.bootcdn.net/ajax/libs/jquery/3.6.0/jquery.min.js"></script>
<script>
    function sendMessage() {
        var text = $("#txt").val()

        // 基于Ajax将用户输入的文本信息发送到后台（偷偷发请求）。
        $.ajax({
            url: '/send/msg/',
            type: 'GET',
            data: {
                text: text
            }, success: function (res) {
                console.log("请求发送成功", res);
            }
        })
    }

    max_index = 0;
    // 每个2s向后台发送请求获取数据并展现到界面上
    setInterval(function () {
        // 发送请求获取数据
        $.ajax({
            url: '/get/msg/',
            data: {
                index: max_index
            },
            type: 'GET',
            success: function (dataDict) {
                max_index = dataDict.max_index;

                $.each(dataDict.data, function (index, item) {
                    console.log(index, item);
                    // 将内容拼接成div标签，并添加到message区域
                    var tag = $("<div>")
                    tag.text(item)              // <div>item</div>
                    $('#message').append(tag);  // 添加至message区域
                })
            }
        })
    }, 2000)
</script>
</body>
</html>
```

## 长轮询实现实时通信

后端基于队列来实现给请求一个最长等待时间（堵塞），前端采用递归ajax来实现获取返回响应之后立刻再发一个请求

- 访问 /home/ 显示的聊天室界面 + 每个用户创建一个队列
- 点击发送内容，数据也可以发送到后台 + 加载到每个人的队列中
- 获取消息，去自己的队列获取数据，再展示