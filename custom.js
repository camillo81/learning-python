MathJax.Hub.Config({
  TeX: { 
         Macros: { RR: "{RRRRR}",
                   bold: ["{\\boldsymbol #1}",1]
                 },
         equationNumbers: { autoNumber: "AMS" },
         TagSide: "left",
         MultLineWidth: "85%",
         extensions: ["autoload-all.js"]
       },
  jax: ["input/TeX","output/HTML-CSS"],
  displayAlign: "left" 
});
