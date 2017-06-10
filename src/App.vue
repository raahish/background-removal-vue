<template>
  <div id="app">
    <div class="container">
      <div class="text-center">
        <a href="/" title="GreenScreen - Cut your images with AI">
          <img src="./assets/logo-with-text.svg" width="50%" alt="GreenScreen - Cut your images with AI "/>
        </a>
      </div>
    </div>
    <div class="navbar navbar-inverse">
      <div class="container">
        <ul class="nav navbar-nav">
          <li class="navbar-text">using Keras.JS</li>
          <li><a class="navbar-link" href="https://medium.com/p/35648f9dc5fb">Blog Post</a></li>
          <li><a class="navbar-link" href="https://greenscreen-ai.boorgle.com/">Server-side solution</a></li>
        </ul>
      </div>
    </div>
    <router-view></router-view>
  </div>
</template>

<script>
export default {
  name: 'app'
}
</script>

<style>
  @import url('//fonts.googleapis.com/css?family=Open+Sans:400,700,800|Open+Sans+Condensed:300,700');
  @import url('//maxcdn.bootstrapcdn.com/bootstrap/3.3.7/css/bootstrap.min.css');

  body {
    font-family: "Open Sans", "Helvetica Neue", Helvetica, Arial, sans-serif;
    display: flex;
    flex-direction: column;
    min-height: 100vh;
  }

  #app {
    display: flex;
    flex-direction: column;
    flex: 1;
  }
  #content {
    flex: 1;
  }

  .container {
    max-width: 960px;
  }


</style>
