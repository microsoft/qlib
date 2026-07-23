"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var tslib_1 = require("tslib");
var echarts = tslib_1.__importStar(require("echarts"));
var core_1 = tslib_1.__importDefault(require("./core"));
// export the Component the echarts Object.
var EChartsReact = /** @class */ (function (_super) {
    tslib_1.__extends(EChartsReact, _super);
    function EChartsReact(props) {
        var _this = _super.call(this, props) || this;
        // 初始化为 echarts 整个包
        _this.echarts = echarts;
        return _this;
    }
    return EChartsReact;
}(core_1.default));
exports.default = EChartsReact;
//# sourceMappingURL=index.js.map