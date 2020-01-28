var GRID_COLOR = 'rgba(255,255,255,.1)';
var TICK_FONT_COLOR = 'rgba(255,255,255,.75)';
var H_LINE_COLOR = 'rgba(255,255,255,.4)';
var V_LINE_COLOR = 'rgba(255,255,255,.1)';

var trace1 =  {
      type: 'scatter',
      marker : { symbol: 'square-open-dot', size:10  },
      xaxis: "x1",
      yaxis: "y1",
      mode: 'lines+markers'
};

var trace2 = { 
   mode: 'lines',
   type: 'scatter',  
   yaxis: "x2",
   line: {
      color: 'rgba(213,42,224,.5)' 
   }
};

var layout = {
   title: { 
      font: {  color:'#ffffff' },
      xref: 'x',   
      yref: 'y',  
      x: 0.05, 
   },
   height: 517,
   width: 937,
   paper_bgcolor: "rgba(9,29,63,1)",  // For exporting in PNG!
   plot_bgcolor: "rgba(9,29,63,1)",   // For exporting in PNG!
   xaxis:{
         //autorange: true,
         zerolinecolor: H_LINE_COLOR, 
         zerolinewidth: 1,
         gridcolor: GRID_COLOR, 
         linecolor: H_LINE_COLOR,  
         title: { font: { size: 15, color: '#b1b1b1' } },
         tickfont: { color: TICK_FONT_COLOR},
         autorange: "reversed" 
   },
   yaxis:{
         zerolinecolor: V_LINE_COLOR, 
         zerolinewidth: 1, 
         gridcolor: GRID_COLOR, 
         linecolor: V_LINE_COLOR,
         title: { font: {  size: 15, color: '#b1b1b1' }}, 
         tickfont: { color: TICK_FONT_COLOR},
         scaleanchor: "x",
         scaleratio: 1,
         autorange: "reversed" 
   },
   showlegend: false }; 
   
// Create all Colors (same than on the canvas)
var rainbow = new Rainbow();
rainbow.setNumberRange(0, 255);

var all_colors = [];
var total = all_data.x1_vals.length; 
var step = parseInt(255/total);  

for (var i = 0; i <= 255; i = i + step) {
   all_colors.push('rgba('+hexToRgb(rainbow.colourAt(i))+')'); 
}

// We had the color scale for X
trace1.marker.color  =  all_colors; 