declare const visualDefault: {
    /**
     * @public
     */
    get: (visualType: string, key: 'active' | 'inactive', isCategory?: boolean) => string | number | string[] | number[];
};
export default visualDefault;
