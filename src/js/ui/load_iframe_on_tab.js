// Load iframe on shown tab only 
$('a.load-frames-on-active').on('click', function(event){
   $($(this).attr('href')).find('iframe').each(function(){
      if($(this).attr('src')=='' || typeof $(this).attr('src')=='undefined') {
         $(this).attr('src',$(this).attr('data-src'));
         $(this).parent('.load_if').addClass('d');
      }
   });
});