declare const _default: {
    darkMode: string;
    colorBy: string;
    color: string[];
    gradientColor: string[];
    aria: {
        decal: {
            decals: ({
                color: string;
                dashArrayX: number[];
                dashArrayY: number[];
                symbolSize: number;
                rotation: number;
                symbol?: undefined;
            } | {
                color: string;
                symbol: string;
                dashArrayX: number[][];
                dashArrayY: number[];
                symbolSize: number;
                rotation?: undefined;
            } | {
                color: string;
                dashArrayX: number[];
                dashArrayY: number[];
                rotation: number;
                symbolSize?: undefined;
                symbol?: undefined;
            } | {
                color: string;
                dashArrayX: number[][];
                dashArrayY: number[];
                symbolSize?: undefined;
                rotation?: undefined;
                symbol?: undefined;
            } | {
                color: string;
                dashArrayX: number[][];
                dashArrayY: number[];
                rotation: number;
                symbolSize?: undefined;
                symbol?: undefined;
            })[];
        };
    };
    textStyle: {
        fontFamily: string;
        fontSize: number;
        fontStyle: string;
        fontWeight: string;
    };
    blendMode: any;
    stateAnimation: {
        duration: number;
        easing: string;
    };
    animation: string;
    animationDuration: number;
    animationDurationUpdate: number;
    animationEasing: string;
    animationEasingUpdate: string;
    animationThreshold: number;
    progressiveThreshold: number;
    progressive: number;
    hoverLayerThreshold: number;
    useUTC: boolean;
};
export default _default;
