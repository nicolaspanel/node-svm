'use strict';



function percent(Handlebars) {
    Handlebars.registerHelper('percent', function (val) {
        return Math.round(parseFloat(val)*1e4)/1e2+'%';
    });
}

module.exports = percent;