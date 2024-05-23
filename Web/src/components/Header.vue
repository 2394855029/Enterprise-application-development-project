<template>
    <!--显示logo和slogan-->
    <div class="header">
        <img src="@/assets/ChatGPT.png" alt="logo" @click="to_index"/>
        <span @click="to_index">ChatPaper</span>
        </div>  
</template>

<script>
export default {
  name: 'Header',
  methods: {
    to_index() {
      this.$router.push({ path: '/' });
    },
  }
}
</script>

<style>
.header{
  display: flex;
  /* 左对齐 */
  justify-content: flex-start;
  align-items: center;
  padding-left: 25px;
  font-size: 30px;
  font-weight: bold;
  color: #409eff;
  /* background-color: #d3e3fd; */
  img{
    width: 36px;
    height: 36px;
    margin-right: 6px;
  }
}
</style>