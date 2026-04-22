var Pattern = (function () {
    function Pattern(image, repeat) {
        this.image = image;
        this.repeat = repeat;
        this.x = 0;
        this.y = 0;
        this.rotation = 0;
        this.scaleX = 1;
        this.scaleY = 1;
    }
    return Pattern;
}());
export default Pattern;
