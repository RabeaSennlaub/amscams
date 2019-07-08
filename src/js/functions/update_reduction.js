function update_reduction_on_canvas_and_table(json_resp) {
    var smf = json_resp['meteor_frame_data'];

    if(typeof smf == 'undefined') {
        smf = json_resp['sd_meteor_frame_data'];
    }

    var lc = 0;
    var table_tbody_html = '';
    var rad = 6;


    var all_frame_ids = [];

    // Get all the frame IDs so we know which one are missing
    $.each(smf, function(i,v){
        all_frame_ids.push(parseInt(v[1]));
    });

    // Create Colors
    var rainbow = new Rainbow();
    var all_colors = [];
    var total = all_frame_ids.length; 
    var step = parseInt(255/total);
    for (var i = 0; i <= 255; i = i + step) {
        all_colors.push(rainbow.colourAt(i));
    }

    console.log(all_colors);
    
    $.each(smf, function(i,v){
 
        
        // Get thumb path
        var frame_id = parseInt(v[1]);
        var thumb_path = my_image.substring(0,my_image.indexOf('-half')) + '-frm' + frame_id + '.png';
        var square_size = 6;
        var _time = v[0].split(' ');
  
        // Thumb	#	Time	X/Y - W/H	Max PX	RA/DEC	AZ/EL
        table_tbody_html+= '<tr id="fr_'+frame_id+'" data-org-x="'+v[2]+'" data-org-y="'+v[3]+'"><td><img alt="Thumb #'+frame_id+'" src='+thumb_path+' width=50 height=50 class="img-fluid select_meteor"/></td>';
        table_tbody_html+= '<td>'+frame_id+'</td><td>'+_time[1]+'</td><td>'+v[7]+'&deg;/'+v[8]+'&deg;</td><td>'+v[9]+'&deg;/'+v[10]+'&deg;</td><td>'+ parseFloat(v[2])+'/'+parseFloat(v[3]) +'</td><td>'+ v[4]+'x'+v[5]+'</td>';
        table_tbody_html+= '<td>'+v[6]+'</td>';
        table_tbody_html+= '<td><a class="btn btn-danger btn-sm delete_frame"><i class="icon-delete"></i></a></td>';

        if(i==0) {
            // We add a "+" before and after on if necessary
            table_tbody_html+= '<td class="position-relative"><a class="btn btn-success btn-sm select_meteor"><i class="icon-target"></i></a><a title="Add a frame" class="btn btn-primary btn-sm btn-mm add_f" data-rel="'+ (frame_id-1) +'"><i class="icon-plus"></i></a>';

            if(all_frame_ids.indexOf((frame_id+1))==-1) {
                table_tbody_html+= '<a class="btn btn-primary btn-sm btn-pp add_f" title="Add a frame" data-rel="'+ (frame_id+1) +'"><i class="icon-plus"></i></a></td>';
            } 

            table_tbody_html+= '</td>';

        } else {
            // We add a "+" after only if we don't have the next frame in all_frame_ids
            if(all_frame_ids.indexOf((frame_id+1))==-1) {
                table_tbody_html+= '<td class="position-relative"><a class="btn btn-success btn-sm select_meteor"><i class="icon-target"></i></a><a title="Add a frame" class="btn btn-primary btn-sm btn-pp add_f" data-rel="'+ (frame_id+1) +'"><i class="icon-plus"></i></a></td>';
            } else {
                table_tbody_html+= '<td class="position-relative"><a class="btn btn-success btn-sm select_meteor"><i class="icon-target"></i></a></td>';
            }
        }
    

        // Add Rectangle
        canvas.add(new fabric.Rect({
            fill: 'rgba(0,0,0,0)', 
            strokeWidth: 1, 
            stroke: all_colors[i], //'rgba(230,100,200,.5)', 
            left:  v[2]/2-rad, 
            top:   v[3]/2-rad,
            width: 10,
            height: 10 ,
            selectable: false,
            type: 'reduc_rect'
        }));

    });

    // Replace current table content
    $('#reduc-tab tbody').html(table_tbody_html);

    // Reload the actions
    reduction_table_actions();
}