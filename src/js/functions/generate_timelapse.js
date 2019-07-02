function add_timelapse_modal() {
    $('#timelapse_modal').remove();
 

    $('<div class="modal" tabindex="-1" role="dialog"> \
        <div class="modal-dialog modal-dialog-centered modal-lg" role="document"> \
            <div class="modal-content"> \
            <div class="modal-header"> \
                <h5 class="modal-title">Generate Timelapse</h5> \
                <button type="button" class="close" data-dismiss="modal" aria-label="Close"> \
                <span aria-hidden="true">&times;</span> \
                </button> \
            </div> \
            <div class="modal-body"> \
                <form> \
                    <div class="row"> \
                        <div class="col-sm-6"> \
                            <div class="form-group row mb-1"> \
                                <label class="col-sm-4 col-form-label"><b>Date</b></label> \
                                <div class="col-sm-8"> \
                                    <input type="text" readonly class="form-control-plaintext" id="tl_date" name="tl_date" value=""> \
                                </div> \
                            </div> \
                            <div class="form-group row mb-1"> \
                                <label class="col-sm-4 col-form-label"><b>Cam Id</b></label> \
                                <div class="col-sm-8"> \
                                    <input type="text" readonly class="form-control-plaintext" id="tl_cam_id" name="tl_cam_id" value=""> \
                                </div> \
                            </div> \
                            <div class="form-group row mb-1"> \
                                <label class="col-sm-4 col-form-label"><b>Frame Count</b></label> \
                                <div class="col-sm-8"> \
                                    <input type="text" readonly class="form-control-plaintext" id="tot_f" value=""> \
                                </div> \
                            </div> \
                            <div class="form-group row mb-1"> \
                                <label class="col-sm-4 col-form-label"><b>Duration</b></label> \
                                <div class="col-sm-8"> \
                                    <input type="text" readonly class="form-control-plaintext" id="tld" value=""> \
                                </div> \
                            </div> \
                        </div> \
                        <div class="col-sm-6"> \
                            <div class="form-group row mb-1"> \
                                <label for="fps" class="col-sm-4 col-form-label"><b>FPS</b></label> \
                                <div class="col-sm-8"> \
                                    <select id="fps" class="form-control"> \
                                        <option value="1">1 fps</option> \
                                        <option value="5">5 fps</option> \
                                        <option value="10" selected>10 fps</option> \
                                        <option value="15">15 fps</option> \
                                        <option value="24.975">24.975 fps</option> \
                                        <option value="29.97">29.97 fps</option> \
                                        <option value="54.94">54.94 fps</option> \
                                    </select> \
                                </div> \
                            </div> \
                            <div class="form-group row mb-1"> \
                                <label for="dim" class="col-sm-4 col-form-label"><b>Dimension</b></label> \
                                <div class="col-sm-8"> \
                                    <select id="dim" class="form-control"> \
                                        <option value="1920:1080">1920x1080</option> \
                                        <option value="1280:720" selected>1280x720</option> \
                                        <option value="640:320">640x320</option> \
                                    </select> \
                                </div> \
                            </div> \
                            <div class="form-group row mb-1"> \
                                <label for="text_pos" class="col-sm-4 col-form-label"><b>Info pos.</b></label> \
                                <div class="col-sm-8"> \
                                    <select id="text_pos" class="form-control"> \
                                        <option value="tr"selected>Top right</option> \
                                        <option value="tl" >Top Left</option> \
                                        <option value="br" >Bottom Right</option> \
                                        <option value="bl" >Bottom Left</option> \
                                    </select> \
                                </div> \
                            </div> \
                            <div class="form-group row mb-1"> \
                                <label for="wat_pos" class="col-sm-4 col-form-label"><b>Logo pos.</b></label> \
                                <div class="col-sm-8"> \
                                    <select id="wat_pos" class="form-control"> \
                                        <option value="tr" >Top right</option> \
                                        <option value="tl" >Top Left</option> \
                                        <option value="br" >Bottom Right</option> \
                                        <option value="bl" selected>Bottom Left</option> \
                                    </select> \
                                </div> \
                            </div> \
                        </div> \
                </form> \
            </div> \
            <div class="modal-footer"> \
                <button type="button" class="btn btn-primary">Generate</button> \
                <button type="button" class="btn btn-secondary" data-dismiss="modal">Cancel</button> \
            </div> \
            </div> \
        </div> \
        </div>').appendTo('body').modal('show');

    // How many frames 
    hmf = $('img.lz').not('.process').length;
    $('#tot_f').val(hmf);

    // Cam ID 
    $('#tl_cam_id').val($('#cam_id').text());

    // Date
    $('#tl_date').val($('input[name=cur_date]').val());

    // Init duration
    $('#tld').val(parseFloat($('#tot_f').val()/parseFloat($('#fps').val())).toFixed(2) + ' seconds');

    // Update duration 
    $('#fps').unbind('change').bind('change',function() {
        $('#tld').val(parseFloat($('#tot_f').val()/parseFloat($(this).val())).toFixed(2) + ' seconds');
    });
}

$(function() {
    $('#create_night_anim').click(function() {
        add_timelapse_modal();
    });
})