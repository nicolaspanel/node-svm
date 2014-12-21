'use strict';



function percent(Handlebars) {
    Handlebars.registerHelper('number', function (val, precision) {
        precision = precision  || 0;
        return Math.round(parseFloat(val)*Math.pow(10, precision))/Math.pow(10, precision)+'';
    });
}

module.exports = percent;