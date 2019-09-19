// Update selector position and corresponding data
function update_select_preview(top,left,margins,W_factor,H_factor,cursor_dim, cur_step_start,show_pos) {
   
   // Move Selector
   $("#selector").css({
      top: top - cursor_dim/2,
      left: left - cursor_dim/2
   });

   sel_x = Math.floor(left)+margins;
   sel_y = Math.floor(top)+margins;

   if(show_pos) {
      if(cur_step_start) {
         // Update START X/Y
         $('#res .start').html('<b style="color:green">START</b> x:' + Math.floor(sel_x*W_factor)+ 'px ' + 'y:'+  Math.floor(sel_y*H_factor) +'px');
         $('#selector').css('border-color','red');
      } else {
         // Update END X/Y
         $('#res .end').html('<b style="color:red">END</b> x:' + Math.floor(sel_x*W_factor)+ 'px ' + 'y:'+  Math.floor(sel_y*H_factor) +'px');
         $('#selector').css('border-color','green');
      }
   }
  

   return !cur_step_start
}


// Create  select meteor position from stack
function create_meteor_selector_from_stack(image_src) {
   var cursor_dim = 24;            // Cursor dimension
   var margins = 12;                // Max position (x,y) of the meteor inside the cursor

   var real_W = 1920;
   var real_H = 1080;

   var prev_W = 1075;              // Preview
   var prev_H = 605;
     
   var cursor_border_width  = 2; 
   
   var sel_x = prev_W/2-cursor_dim/2;
   var sel_y = prev_H/2-cursor_dim/2;

   var W_factor = real_W/prev_W;
   var H_factor = real_H/prev_H; 

   var cur_step_start = true;
 
   var init_top = prev_H/2-cursor_dim/2;
   var init_left = prev_W/2-cursor_dim/2;

 
   $('<h1>Manual Reduction Step 1</h1>\
     <div class="box">\
     <div class="modal-header p-0" style="border:none!important">\
      <div class="alert alert-info mb-3 p-1 pr-1 pl-2">Select the STARTING point of the meteor path.</div>\
      <div id="res" style="text-align:right"><span class="start"></span><br/><span class="end" ></span></div>\
     </div>\
     <div id="draggable_area" style="width:'+(prev_W+margins*2) + 'px; height:' +( prev_H+margins*2) + 'px;margin:0 auto;">\
     <div id="main_view" style="background-color:#000;background-image:url('+image_src+'); width:'+prev_W+'px; height:'+prev_H+'px; margin: 0 auto; position:relative; background-size: contain;">\
      <div id="selector" class="ng pa" style="width:'+cursor_dim+'px; height:'+cursor_dim+'px; border:'+cursor_border_width+'px solid green;"></div>\
   </div></div>').appendTo($('#step1'));
   
    
   // Default pos
   update_select_preview(init_top,init_left,margins,W_factor,H_factor,cursor_dim,false);
     
   offset = $('#main_view').offset();

   // Move on click
   $('#main_view').click(function(e) {
      var top =  e.pageY - offset.top;
      var left = e.pageX - offset.left;
      cur_step_start = update_select_preview(top,left,margins,W_factor,H_factor,cursor_dim,cur_step_start,true);
      e.stopImmediatePropagation();
      return false;
   });
   

}
