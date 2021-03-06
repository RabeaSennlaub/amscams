 


jQuery.fn.extend({
   loading_button:function() {
      var $btn = $(this);
      var h=$btn.outerHeight(),w=$btn.outerWidth();
      if(typeof $btn.attr('data-init') == 'undefined' || $btn.attr('data-init') == '') {
         $btn.attr('data-init',$btn.html()).attr('style','height:'+h+'px!important;width:'+w+'px!important')
         $btn.html('<img src="/APPS/dist/img/loader.svg" class="img-fluid" style="height:calc('+h+'px - 2*.25rem)"/>');
         $btn.attr('disabled','disabled').addClass('disabled');
      }
   },
   load_done_button: function() {
      $(this).html($(this).attr('data-init')); 
      $(this).attr('data-init','');
      $(this).removeAttr('disabled').removeClass('disabled');
   } 
 });


function hide_bottom_action() {
   $('#bottom_action_bar').addClass('hd').slideUp();
}
function show_bottom_action() {
   $('#bottom_action_bar').removeClass('hd').slideDown();
}



function getDelete() {
   return $('.toDel').length
}

function getConf() {
   return $('.toConf').length
}

function check_bottom_action() {
   var toDel = getDelete(), toConf = getConf();
   if(toDel>0 || toConf >0) {
      show_bottom_action();
      toDel>0?$('#del_text').text('('+toDel+' to delete)'):$('#del_text').text('');
      toConf>0?$('#conf_text').text('('+toConf+' to confirm)'):$('#conf_text').text('');
   } else {
      hide_bottom_action() 
   }
}
 
