/** @type {import('jest').Config} */
module.exports = {
  preset: 'ts-jest',
  testEnvironment: 'node',
  roots: ['<rootDir>/propulsion-controller/src/__tests__'],
  testMatch: ['**/*.test.ts'],
  transform: {
    '^.+\\.tsx?$': ['ts-jest', { tsconfig: '<rootDir>/tsconfig.json' }],
  },
  collectCoverageFrom: [
    'propulsion-controller/src/**/*.ts',
    '!propulsion-controller/src/__tests__/**',
  ],
};
