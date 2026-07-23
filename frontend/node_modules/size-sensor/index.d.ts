declare namespace sizeSensor {
    type StyledElement = Element & ElementCSSInlineStyle;
    export const bind: <T extends StyledElement = HTMLElement>(
        element: T | null,
        cb: (element: T | null) => void
    ) => () => void;
    export const clear: (element: StyledElement | null) => void;
}
export = sizeSensor;
