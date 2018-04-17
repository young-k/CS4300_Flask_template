$(".c_title").trigger("click");

$(".more").toggle(function(){
    $(this).text("less..").siblings(".complete").show();    
}, function(){
    $(this).text("more..").siblings(".complete").hide();    
});

var $el, $ps, $up, totalHeight;
$(document).ready(function(){
	$(".preview .button").click(function() {
	  totalHeight = 0

	  $el = $(this);
	  $p  = $el.parent();
	  $up = $p.parent();
	  $ps = $up.find("p:not('.read-more')");
	  
	  // measure how tall inside should be by adding together heights of all inside paragraphs (except read-more paragraph)
	  $ps.each(function() {
	    totalHeight += $(this).outerHeight() + 20;
	  });
	        
	  $up
	    .css({
	      // Set height to prevent instant jumpdown when max height is removed
	      "height": $up.height(),
	      "max-height": 9999
	    })
	    .animate({
	      "height": totalHeight
	    });
	  
	  // fade out read-more
	  $p.fadeOut();
	  
	  // prevent jump-down
	  return false;
	   
	});
});