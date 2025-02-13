document.addEventListener('DOMContentLoaded', function(){
     const imageHome = document.getElementById("choose_home");
     const imageAway = document.getElementById("choose_away");


     let imageHome_value = imageHome.value
     let imageAway_value = imageAway.value

     let selectedImage2 = imageAway.value;
     let selectedImage = imageHome.value;

     const image_change_home = document.getElementById('doinha')
     const image_change_away = document.getElementById('doikhach')
     
     imageHome.addEventListener('change', function() {
          selectedImage = imageHome.value;
          if (selectedImage != selectedImage2){
               image_change_home.src = selectedImage;
               imageHome_value = imageHome.value
          }
               
          else{
               imageHome.value = imageHome_value
               alert("Vui lòng chọn 2 đội khác nhau")
          }
          });
     
     imageAway.addEventListener('change', function() {
          selectedImage2 = imageAway.value;
          if (selectedImage != selectedImage2){
               image_change_away.src = selectedImage2;
               imageAway_value = imageAway.value
          }
               
          else{
               imageAway.value = imageAway_value
               alert("Vui lòng chọn 2 đội khác nhau")
          }
          });     
})
