// Location: /src/styles/styled.ts
// Base styled-components setup with theme integration

import styled, { createGlobalStyle, css, keyframes } from 'styled-components';
import { motion } from 'framer-motion';

// Global styles
export const GlobalStyles = createGlobalStyle`
  ${({ theme }) => theme.globalStyles}
`;

// Animation keyframes
export const glitchAnimation = keyframes`
  0%, 100% {
    text-shadow: 
      0 0 10px var(--color-primary-80),
      2px 2px 20px var(--color-neural-80),
      -2px -2px 20px var(--color-success-80);
  }
  25% {
    text-shadow: 
      2px 2px 20px var(--color-neural-80),
      -2px -2px 20px var(--color-success-80),
      0 0 10px var(--color-primary-80);
  }
  50% {
    text-shadow: 
      -2px -2px 20px var(--color-success-80),
      0 0 10px var(--color-primary-80),
      2px 2px 20px var(--color-neural-80);
  }
  75% {
    text-shadow: 
      0 0 10px var(--color-primary-80),
      -2px -2px 20px var(--color-success-80),
      2px 2px 20px var(--color-neural-80);
  }
`;

// Base components
export const BaseCard = styled(motion.div)<{ $variant?: 'default' | 'bordered' | 'elevated' }>`
  position: relative;
  background: rgba(17, 24, 39, 0.9);
  backdrop-filter: blur(12px);
  border-radius: var(--radius-xl);
  padding: 1.25rem;
  overflow: hidden;
  
  ${({ $variant = 'default' }) => {
    switch ($variant) {
      case 'bordered':
        return css`
          border: 1px solid var(--color-primary-60);
          box-shadow: 0 0 20px var(--color-primary-20);
        `;
      case 'elevated':
        return css`
          box-shadow: var(--shadow-xl), 0 0 40px var(--color-primary-10);
        `;
      default:
        return css`
          border: 1px solid var(--color-primary-40);
        `;
    }
  }}
  
  transition: all var(--transition-normal);
  
  &:hover {
    transform: translateY(-2px);
    box-shadow: 0 0 30px var(--color-primary-60);
  }
`;

export const GradientText = styled.span<{ $gradient?: string }>`
  background: ${({ $gradient }) => $gradient || 'linear-gradient(to right, var(--color-primary), var(--color-accent), var(--color-neural))'};
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  font-weight: bold;
`;

export const Button = styled(motion.button)<{
  $variant?: 'primary' | 'secondary' | 'danger' | 'success';
  $size?: 'sm' | 'md' | 'lg';
  $fullWidth?: boolean;
}>`
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 0.5rem;
  font-family: inherit;
  font-weight: bold;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  border-radius: var(--radius-lg);
  cursor: pointer;
  transition: all var(--transition-fast);
  border: 2px solid transparent;
  
  ${({ $size = 'md' }) => {
    switch ($size) {
      case 'sm':
        return css`
          padding: 0.5rem 1rem;
          font-size: 0.75rem;
        `;
      case 'lg':
        return css`
          padding: 0.875rem 2rem;
          font-size: 1rem;
        `;
      default:
        return css`
          padding: 0.75rem 1.5rem;
          font-size: 0.875rem;
        `;
    }
  }}
  
  ${({ $fullWidth }) => $fullWidth && css`
    width: 100%;
  `}
  
  ${({ $variant = 'primary' }) => {
    switch ($variant) {
      case 'primary':
        return css`
          background: var(--color-primary-30);
          border-color: var(--color-primary);
          color: var(--color-primary);
          
          &:hover:not(:disabled) {
            background: var(--color-primary-40);
            box-shadow: 0 0 20px var(--color-primary-60);
          }
        `;
      case 'secondary':
        return css`
          background: rgba(75, 85, 99, 0.5);
          border-color: #4B5563;
          color: #9CA3AF;
          
          &:hover:not(:disabled) {
            border-color: #6B7280;
            color: #D1D5DB;
          }
        `;
      case 'danger':
        return css`
          background: var(--color-danger-30);
          border-color: var(--color-danger);
          color: var(--color-danger);
          
          &:hover:not(:disabled) {
            background: var(--color-danger-40);
            box-shadow: 0 0 20px var(--color-danger-60);
          }
        `;
      case 'success':
        return css`
          background: var(--color-success-30);
          border-color: var(--color-success);
          color: var(--color-success);
          
          &:hover:not(:disabled) {
            background: var(--color-success-40);
            box-shadow: 0 0 20px var(--color-success-60);
          }
        `;
    }
  }}
  
  &:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
  
  &:focus-visible {
    outline: 2px solid var(--color-primary);
    outline-offset: 2px;
  }
`;

export const Input = styled.input<{ $hasError?: boolean }>`
  width: 100%;
  padding: 0.75rem 1rem;
  background: rgba(17, 24, 39, 0.8);
  border: 2px solid ${({ $hasError }) => $hasError ? 'var(--color-danger)' : 'var(--color-primary-40)'};
  border-radius: var(--radius-lg);
  color: var(--color-accent);
  font-family: inherit;
  font-size: 0.875rem;
  transition: all var(--transition-fast);
  
  &::placeholder {
    color: #6B7280;
  }
  
  &:focus {
    outline: none;
    border-color: var(--color-primary);
    background: rgba(17, 24, 39, 0.95);
    box-shadow: 0 0 0 4px var(--color-primary-20);
  }
  
  &:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
`;

export const Label = styled.label`
  display: block;
  margin-bottom: 0.5rem;
  font-size: 0.875rem;
  font-weight: 600;
  color: #9CA3AF;
  text-transform: uppercase;
  letter-spacing: 0.05em;
`;

// Layout components
export const PageContainer = styled.div`
  min-height: 100vh;
  background: linear-gradient(to bottom right, #030712, #111827, #000000);
  position: relative;
  overflow: hidden;
`;

export const BackgroundEffects = styled.div`
  position: fixed;
  inset: 0;
  opacity: 0.2;
  pointer-events: none;
  
  &::before {
    content: '';
    position: absolute;
    inset: 0;
    background-image: 
      radial-gradient(circle at 20% 50%, var(--color-primary-20) 0%, transparent 50%),
      radial-gradient(circle at 80% 80%, var(--color-neural-20) 0%, transparent 50%),
      radial-gradient(circle at 40% 20%, var(--color-success-20) 0%, transparent 50%);
  }
`;

export const Grid = styled.div<{ $cols?: number; $gap?: string }>`
  display: grid;
  grid-template-columns: repeat(${({ $cols = 1 }) => $cols}, 1fr);
  gap: ${({ $gap = '1.5rem' }) => $gap};
  
  @media (max-width: 1280px) {
    grid-template-columns: repeat(${({ $cols = 1 }) => Math.min($cols, 3)}, 1fr);
  }
  
  @media (max-width: 768px) {
    grid-template-columns: repeat(${({ $cols = 1 }) => Math.min($cols, 2)}, 1fr);
  }
  
  @media (max-width: 640px) {
    grid-template-columns: 1fr;
  }
`;

export const Flex = styled.div<{
  $direction?: 'row' | 'column';
  $align?: string;
  $justify?: string;
  $gap?: string;
  $wrap?: boolean;
}>`
  display: flex;
  flex-direction: ${({ $direction = 'row' }) => $direction};
  align-items: ${({ $align = 'stretch' }) => $align};
  justify-content: ${({ $justify = 'flex-start' }) => $justify};
  gap: ${({ $gap = '0' }) => $gap};
  ${({ $wrap }) => $wrap && 'flex-wrap: wrap;'}
`;
