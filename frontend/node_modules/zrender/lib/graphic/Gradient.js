var Gradient = (function () {
    function Gradient(colorStops) {
        this.colorStops = colorStops || [];
    }
    Gradient.prototype.addColorStop = function (offset, color) {
        this.colorStops.push({
            offset: offset,
            color: color
        });
    };
    return Gradient;
}());
export default Gradient;
