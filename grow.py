import { OtpVerification } from '@/components/OtpVerification';

const Index = () => {
  const handleVerifySuccess = (otp: string) => {
    console.log('OTP verified successfully:', otp);
    // Here you would typically redirect to the next page or update the app state
  };

  const handleResendOtp = () => {
    console.log('Resending OTP...');
    // Here you would make an API call to resend the OTP
  };

  const handleChangeContact = () => {
    console.log('Change contact info...');
    // Here you would navigate back to contact input or show a modal
  };

  return (
    <OtpVerification
      phoneNumber="+1 (555) 123-4567"
      onVerifySuccess={handleVerifySuccess}
      onResendOtp={handleResendOtp}
      onChangeContact={handleChangeContact}
    />
  );
};

export default Index;